"""
Experiment 9B Tuned2: Domain-Targeted Causal Features + 3-Seed Tuned 9B Bagging
Builds on main9B_tuned1.py by adding:
  - SIM vs Device vs Account age ratios (separates sim_swap vs account_takeover)
  - All-time causal device sharing & sender device count
  - Consecutive-tx city change & impossible travel (<1h city jump)
  - Per-account historical baselines (ip_risk_vs_acc_mean, login_sec_vs_acc_mean)
  - Whole-rupee / round-50 flags + balance_after drain signal
  - 1h receiver burst fan-in, receiver inter-arrival time, & amt_vs_merch_mean
  - Cross-midnight P2P inflow-to-outflow layering fix
  - 3-seed bagging + finer Macro-F1 multiplier optimization

Run:
    python main9B_tuned2.py
"""

import gc
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 1. SETTINGS (Update column names here to match your CSVs)
# ==============================================================================
DATA_DIR = "./data"
ID_COL = "txn_id"
TARGET_COL = "label"
TIME_COL = "timestamp"
AMOUNT_COL = "amount"
ACC_COL = "sender_id"
MERCH_COL = "receiver_id"

# Validation: "time" (auto-splits final month, e.g. Sep) or "kfold" (5-fold Stratified)
CV_MODE = "time"
N_FOLDS = 5
SEEDS = (42, 1024, 2025)
MODEL_DIR = "./saved_model_9B_tuned2"
SUBMISSION_PATH = "submission_9B_tuned2.csv"


# ==============================================================================
# 2. LOAD & JOIN DATA
# ==============================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=[TIME_COL])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=[TIME_COL])
    raw_accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv").drop_duplicates(subset="account_id")
    merchants = pd.read_csv(f"{DATA_DIR}/merchants.csv").drop_duplicates(subset="merchant_id")

    # Prefix dimension columns so sender, receiver (P2P), and merchant attributes stay distinct
    accounts = raw_accounts.add_prefix("acc_").rename(columns={"acc_account_id": ACC_COL})
    recv_accounts = raw_accounts.add_prefix("recv_").rename(columns={"recv_account_id": MERCH_COL})
    merchants = merchants.add_prefix("merch_").rename(columns={"merch_merchant_id": MERCH_COL})

    train = (
        train.merge(accounts, on=ACC_COL, how="left")
        .merge(recv_accounts, on=MERCH_COL, how="left")
        .merge(merchants, on=MERCH_COL, how="left")
    )
    test = (
        test.merge(accounts, on=ACC_COL, how="left")
        .merge(recv_accounts, on=MERCH_COL, how="left")
        .merge(merchants, on=MERCH_COL, how="left")
    )

    del raw_accounts, accounts, recv_accounts, merchants
    gc.collect()
    return train, test


# ==============================================================================
# 3. FAST VECTORIZED FEATURES (No leakage, no slow per-group rolling loops)
# ==============================================================================
def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    print("Building features...")
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test[TARGET_COL] = np.nan

    # Combine & sort globally by timestamp so EVERY entity group (account, merchant, device)
    # is strictly chronological and Oct rows can only see past history (zero leakage)
    df = pd.concat([train, test], ignore_index=True)
    df = df.sort_values(TIME_COL).reset_index(drop=True)

    # --- Basic time & amount features ---
    df["hour"] = df[TIME_COL].dt.hour.astype("int8")
    df["dayofweek"] = df[TIME_COL].dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["dayofweek"] >= 5).astype("int8")
    df["log_amount"] = np.log1p(df[AMOUNT_COL].clip(lower=0)).astype("float32")
    df["is_whole_rupee"] = ((df[AMOUNT_COL] % 1.0) == 0).astype("int8")
    df["is_round_50"] = ((df[AMOUNT_COL] % 50.0) == 0).astype("int8")
    df["is_round_100"] = ((df[AMOUNT_COL] % 100.0) == 0).astype("int8")

    # Domain signals from transaction & joined account/merchant columns
    if "balance_before" in df.columns:
        df["balance_after"] = (df["balance_before"] - df[AMOUNT_COL]).astype("float32")
        df["amt_to_balance_ratio"] = (
            df[AMOUNT_COL] / (df["balance_before"].clip(lower=0) + 1.0)
        ).astype("float32")

    if "acc_home_city" in df.columns and "device_city" in df.columns:
        df["is_city_mismatch"] = (df["device_city"] != df["acc_home_city"]).astype("int8")

    if "merch_onboarded_on" in df.columns:
        merch_onboarded = pd.to_datetime(df["merch_onboarded_on"], errors="coerce")
        df["merch_age_days"] = ((df[TIME_COL] - merch_onboarded).dt.total_seconds() / 86400.0).astype("float32")

    if "merch_city" in df.columns and "device_city" in df.columns:
        merch_city_mask = df["merch_city"].notna()
        df["is_merch_city_mismatch"] = np.where(
            merch_city_mask, (df["device_city"] != df["merch_city"]).astype("float32"), np.nan
        ).astype("float32")

    # SIM vs Device vs Account age ratios (separates sim_swap vs account_takeover)
    if "sim_age_hours" in df.columns and "sender_account_age_days" in df.columns:
        df["sim_to_acc_age_ratio"] = (
            (df["sim_age_hours"] / 24.0) / (df["sender_account_age_days"].clip(lower=0) + 1.0)
        ).astype("float32")

    if "sim_age_hours" in df.columns and "device_age_days" in df.columns:
        df["sim_to_device_age_ratio"] = (
            (df["sim_age_hours"] / 24.0) / (df["device_age_days"].clip(lower=0) + 1.0)
        ).astype("float32")

    if "device_age_days" in df.columns and "sender_account_age_days" in df.columns:
        df["device_to_acc_age_ratio"] = (
            df["device_age_days"].clip(lower=0) / (df["sender_account_age_days"].clip(lower=0) + 1.0)
        ).astype("float32")

    # --- P2P receiver profile & sender-receiver relationship flags (NaN preserved for merchants) ---
    if "recv_opened_on" in df.columns:
        recv_opened = pd.to_datetime(df["recv_opened_on"], errors="coerce")
        df["recv_account_age_days"] = ((df[TIME_COL] - recv_opened).dt.total_seconds() / 86400.0).astype("float32")

    if "acc_bank_code" in df.columns and "recv_bank_code" in df.columns:
        p2p_mask = df["recv_bank_code"].notna()
        df["is_same_bank_sender_recv"] = np.where(
            p2p_mask, (df["acc_bank_code"] == df["recv_bank_code"]).astype("float32"), np.nan
        ).astype("float32")

    if "acc_home_city" in df.columns and "recv_home_city" in df.columns:
        p2p_city_mask = df["recv_home_city"].notna()
        df["is_same_home_city_sender_recv"] = np.where(
            p2p_city_mask, (df["acc_home_city"] == df["recv_home_city"]).astype("float32"), np.nan
        ).astype("float32")
        if "device_city" in df.columns:
            df["is_device_in_recv_city"] = np.where(
                p2p_city_mask, (df["device_city"] == df["recv_home_city"]).astype("float32"), np.nan
            ).astype("float32")

    # --- Account velocity, spend & behavioral baselines (strictly causal, vectorized) ---
    g_acc = df.groupby(ACC_COL, sort=False)
    df["sec_since_prev_tx"] = g_acc[TIME_COL].diff().dt.total_seconds().fillna(-1).astype("float32")
    df["acc_prior_tx_cnt"] = g_acc.cumcount().astype("int32")

    prior_cnt_nz = df["acc_prior_tx_cnt"].replace(0, np.nan)
    prior_sum = (g_acc[AMOUNT_COL].cumsum() - df[AMOUNT_COL]).astype("float32")
    prior_mean = prior_sum / prior_cnt_nz
    df["amt_vs_acc_mean"] = (df[AMOUNT_COL] / (prior_mean.clip(lower=0) + 1.0)).astype("float32")

    if "ip_risk_score" in df.columns:
        prior_ip_sum = (g_acc["ip_risk_score"].cumsum() - df["ip_risk_score"]).astype("float32")
        prior_ip_mean = prior_ip_sum / prior_cnt_nz
        df["ip_risk_vs_acc_mean"] = (df["ip_risk_score"] - prior_ip_mean).astype("float32")

    if "login_to_txn_sec" in df.columns:
        prior_login_sum = (g_acc["login_to_txn_sec"].cumsum() - df["login_to_txn_sec"]).astype("float32")
        prior_login_mean = prior_login_sum / prior_cnt_nz
        df["login_sec_vs_acc_mean"] = (
            df["login_to_txn_sec"] / (prior_login_mean.clip(lower=0) + 1.0)
        ).astype("float32")

    if "device_city" in df.columns:
        prev_city = g_acc["device_city"].shift(1)
        has_prev_city = prev_city.notna() & df["device_city"].notna()
        df["is_city_changed_prev_tx"] = np.where(
            has_prev_city, (df["device_city"] != prev_city).astype("float32"), np.nan
        ).astype("float32")
        df["is_impossible_travel"] = (
            (df["is_city_changed_prev_tx"] == 1.0)
            & (df["sec_since_prev_tx"] >= 0)
            & (df["sec_since_prev_tx"] < 3600)
        ).astype("int8")

    # Fast burst velocity using floored timestamps (100x faster than rolling)
    min5_bucket = df[TIME_COL].dt.floor("5min")
    hour_bucket = df[TIME_COL].dt.floor("1h")
    day_bucket = df[TIME_COL].dt.floor("1D")

    df["acc_tx_in_5min"] = df.groupby([ACC_COL, min5_bucket], sort=False).cumcount().astype("int32")
    df["acc_tx_in_hour"] = df.groupby([ACC_COL, hour_bucket], sort=False).cumcount().astype("int32")
    df["acc_tx_in_day"] = df.groupby([ACC_COL, day_bucket], sort=False).cumcount().astype("int32")

    g_acc_day = df.groupby([ACC_COL, day_bucket], sort=False)
    df["acc_amt_in_day"] = (g_acc_day[AMOUNT_COL].cumsum() - df[AMOUNT_COL]).astype("float32")

    # Rapid payee switching in 1h (strictly causal distinct receivers per sender per hour)
    df["_first_acc_recv_hour"] = (
        df.groupby([ACC_COL, MERCH_COL, hour_bucket], dropna=False, sort=False).cumcount() == 0
    ).astype("int8")
    df["acc_unique_receivers_in_hour"] = (
        df.groupby([ACC_COL, hour_bucket], sort=False)["_first_acc_recv_hour"].cumsum()
        - df["_first_acc_recv_hour"]
    ).astype("int32")

    # --- Receiver / Merchant aggregations & ticket baselines ---
    g_merch = df.groupby(MERCH_COL, dropna=False, sort=False)
    df["recv_sec_since_prev_tx"] = g_merch[TIME_COL].diff().dt.total_seconds().fillna(-1).astype("float32")
    df["merch_prior_tx_cnt"] = g_merch.cumcount().astype("int32")

    merch_prior_sum = (g_merch[AMOUNT_COL].cumsum() - df[AMOUNT_COL]).astype("float32")
    merch_prior_mean = merch_prior_sum / df["merch_prior_tx_cnt"].replace(0, np.nan)
    df["amt_vs_merch_mean"] = (df[AMOUNT_COL] / (merch_prior_mean.clip(lower=0) + 1.0)).astype("float32")

    df["acc_merch_prior_cnt"] = (
        df.groupby([ACC_COL, MERCH_COL], dropna=False, sort=False).cumcount().astype("int32")
    )
    df["acc_merch_tx_in_day"] = (
        df.groupby([ACC_COL, MERCH_COL, day_bucket], dropna=False, sort=False).cumcount().astype("int32")
    )
    acc_merch_amt_day = (
        df.groupby([ACC_COL, MERCH_COL, day_bucket], dropna=False, sort=False)[AMOUNT_COL].cumsum()
        - df[AMOUNT_COL]
    ).astype("float32")
    df["acc_merch_amt_share_day"] = (
        acc_merch_amt_day / (df["acc_amt_in_day"].clip(lower=0) + 1.0)
    ).astype("float32")

    df["acc_same_amt_in_day"] = (
        df.groupby([ACC_COL, AMOUNT_COL, day_bucket], sort=False).cumcount().astype("int32")
    )
    df["merch_tx_in_hour"] = (
        df.groupby([MERCH_COL, hour_bucket], dropna=False, sort=False).cumcount().astype("int32")
    )
    df["merch_tx_in_day"] = (
        df.groupby([MERCH_COL, day_bucket], dropna=False, sort=False).cumcount().astype("int32")
    )

    # Receiver incoming rupee velocity & distinct-sender fan-in (1h + 1D)
    df["recv_amt_in_hour"] = (
        df.groupby([MERCH_COL, hour_bucket], dropna=False, sort=False)[AMOUNT_COL].cumsum()
        - df[AMOUNT_COL]
    ).astype("float32")
    df["recv_amt_in_day"] = (
        df.groupby([MERCH_COL, day_bucket], dropna=False, sort=False)[AMOUNT_COL].cumsum()
        - df[AMOUNT_COL]
    ).astype("float32")

    df["recv_unique_senders_in_hour"] = (
        df.groupby([MERCH_COL, hour_bucket], dropna=False, sort=False)["_first_acc_recv_hour"].cumsum()
        - df["_first_acc_recv_hour"]
    ).astype("int32")
    df.drop(columns=["_first_acc_recv_hour"], inplace=True)

    df["_first_pair_today"] = (
        df.groupby([ACC_COL, MERCH_COL, day_bucket], dropna=False, sort=False).cumcount() == 0
    ).astype("int8")
    df["recv_unique_senders_in_day"] = (
        df.groupby([MERCH_COL, day_bucket], dropna=False, sort=False)["_first_pair_today"].cumsum()
        - df["_first_pair_today"]
    ).astype("int32")
    df.drop(columns=["_first_pair_today"], inplace=True)

    # Repeat concentration per sender at receiver (merchant collusion signal)
    df["merch_tx_per_sender_in_day"] = (
        df["merch_tx_in_day"] / (df["recv_unique_senders_in_day"] + 1.0)
    ).astype("float32")

    # Sender inflow-to-outflow layering signals (cross-midnight matching via by=ACC_COL)
    inflow_src = df[df["receiver_type"] == "P2P"] if "receiver_type" in df.columns else df
    inflow_src = inflow_src[inflow_src[TIME_COL].notna()]
    if len(inflow_src) > 0:
        inflows = pd.DataFrame({
            ACC_COL: inflow_src[MERCH_COL].reset_index(drop=True),
            "_recv_time": inflow_src[TIME_COL].reset_index(drop=True),
            "_recv_amt": inflow_src[AMOUNT_COL].reset_index(drop=True),
            "_recv_day_total": (inflow_src["recv_amt_in_day"] + inflow_src[AMOUNT_COL]).reset_index(drop=True),
        })
        left_lookup = df[[ACC_COL, TIME_COL]]
        matched = pd.merge_asof(
            left_lookup,
            inflows,
            left_on=TIME_COL,
            right_on="_recv_time",
            by=ACC_COL,
            direction="backward",
            allow_exact_matches=False,
        )
        df["sec_since_sender_last_received"] = (
            (df[TIME_COL] - matched["_recv_time"]).dt.total_seconds().fillna(-1.0).astype("float32")
        )
        same_day_inflow = matched["_recv_time"].dt.floor("1D") == day_bucket
        df["sender_received_amt_in_day"] = np.where(
            same_day_inflow, matched["_recv_day_total"].fillna(0.0), 0.0
        ).astype("float32")
        df["outflow_to_inflow_ratio_day"] = (
            df[AMOUNT_COL] / (df["sender_received_amt_in_day"] + 1.0)
        ).astype("float32")
        df["amt_vs_last_received_ratio"] = (
            df[AMOUNT_COL] / (matched["_recv_amt"].fillna(0.0) + 1.0)
        ).astype("float32")

    if "device_id" in df.columns:
        df["device_tx_in_hour"] = df.groupby(["device_id", hour_bucket], sort=False).cumcount().astype("int32")
        df["device_tx_in_day"] = df.groupby(["device_id", day_bucket], sort=False).cumcount().astype("int32")

        # Causal device sharing (1D + all-time prior) & sender distinct prior devices
        df["_first_dev_sender_today"] = (
            df.groupby(["device_id", ACC_COL, day_bucket], sort=False).cumcount() == 0
        ).astype("int8")
        df["device_unique_senders_in_day"] = (
            df.groupby(["device_id", day_bucket], sort=False)["_first_dev_sender_today"].cumsum()
            - df["_first_dev_sender_today"]
        ).astype("int32")
        df.drop(columns=["_first_dev_sender_today"], inplace=True)

        df["_first_dev_sender_all"] = (
            df.groupby(["device_id", ACC_COL], sort=False).cumcount() == 0
        ).astype("int8")
        df["device_unique_senders_prior"] = (
            df.groupby("device_id", sort=False)["_first_dev_sender_all"].cumsum()
            - df["_first_dev_sender_all"]
        ).astype("int32")
        df["acc_unique_devices_prior"] = (
            df.groupby(ACC_COL, sort=False)["_first_dev_sender_all"].cumsum()
            - df["_first_dev_sender_all"]
        ).astype("int32")
        df.drop(columns=["_first_dev_sender_all"], inplace=True)

        # New device for sender (account takeover signal)
        df["acc_device_prior_cnt"] = df.groupby([ACC_COL, "device_id"], sort=False).cumcount().astype("int32")
        df["is_new_device_for_sender"] = (df["acc_device_prior_cnt"] == 0).astype("int8")

    # --- Prepare feature list & LightGBM categoricals ---
    drop_cols = {
        ID_COL,
        TARGET_COL,
        TIME_COL,
        "_is_train",
        ACC_COL,
        MERCH_COL,
        "device_id",
        "acc_opened_on",
        "merch_onboarded_on",
        "recv_opened_on",
        "recv_home_city",
        "acc_home_city",
        "device_city",
        "merch_city",
        "acc_bank_code",
        "recv_bank_code",
        "app_version",
    }
    feature_cols = [c for c in df.columns if c not in drop_cols]

    for col in feature_cols:
        if df[col].dtype == "object" or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].fillna("MISSING").astype("category")

    train_df = df[df["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_df = df[df["_is_train"] == 0].drop(columns=["_is_train", TARGET_COL]).reset_index(drop=True)

    print(f"Features ready: {len(feature_cols)} columns")
    return train_df, test_df, feature_cols


# ==============================================================================
# 4. TRAIN & VALIDATE LIGHTGBM
# ==============================================================================
def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Vectorized Macro-F1 via bincount confusion matrix (orders of magnitude faster on 500k+ rows)."""
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1s = np.where(denom > 0, (2.0 * tp) / np.maximum(denom, 1), 0.0)
    return float(f1s.mean())


def optimize_class_multipliers(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_passes: int = 6,
) -> np.ndarray:
    """
    Coordinate ascent on per-class probability multipliers w so argmax(probs * w)
    directly maximizes Macro-F1. Tests both uniform and inverse-prior initializations
    to avoid getting trapped in a high-legit local optimum.
    """
    num_classes = probs.shape[1]
    counts = np.bincount(y_true, minlength=num_classes)
    prior_init = (counts.max() / np.maximum(counts, 1)) ** 0.35
    prior_init /= prior_init.mean()

    grid = np.geomspace(0.005, 100.0, 65)
    best_overall_w = np.ones(num_classes, dtype="float64")
    best_overall_f1 = fast_macro_f1(y_true, np.argmax(probs, axis=1), num_classes)

    for init_w in (np.ones(num_classes, dtype="float64"), prior_init):
        weights = init_w.copy()
        current_f1 = fast_macro_f1(y_true, np.argmax(probs * weights, axis=1), num_classes)

        for _ in range(n_passes):
            for c in range(num_classes):
                other_scores = probs * weights
                other_scores[:, c] = -1.0
                best_other_idx = np.argmax(other_scores, axis=1)
                best_other_val = other_scores[np.arange(len(y_true)), best_other_idx]

                p_c = probs[:, c]
                best_m = weights[c]
                for m in grid:
                    preds = np.where(p_c * m > best_other_val, c, best_other_idx)
                    score = fast_macro_f1(y_true, preds, num_classes)
                    if score > current_f1:
                        current_f1 = score
                        best_m = m
                weights[c] = best_m
            weights /= weights.mean()

        if current_f1 > best_overall_f1:
            best_overall_f1 = current_f1
            best_overall_w = weights.copy()

    return best_overall_w.astype("float32")


def print_top_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: np.ndarray) -> None:
    """Print where each class's errors are going (top 3 confused predictions per class)."""
    num_classes = len(class_names)
    cm = np.bincount(y_true * num_classes + y_pred, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    )
    print("\nTop confusions per true class:")
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total == 0:
            continue
        errors = [(j, cm[i, j]) for j in range(num_classes) if j != i and cm[i, j] > 0]
        errors.sort(key=lambda x: x[1], reverse=True)
        top_str = ", ".join(f"{class_names[j]}={cnt} ({100 * cnt / total:.1f}%)" for j, cnt in errors[:3])
        correct_pct = 100.0 * cm[i, i] / total
        print(f"  {name:20s} ({total:6,d} | recall={correct_pct:5.1f}%): {top_str or 'none'}")


def train_and_predict(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[np.ndarray, LabelEncoder, list[lgb.Booster], np.ndarray]:
    train_df = train_df.dropna(subset=[TARGET_COL]).reset_index(drop=True)

    le = LabelEncoder()
    y = le.fit_transform(train_df[TARGET_COL].astype(str))
    num_classes = len(le.classes_)
    print(f"Classes ({num_classes}): {list(le.classes_)}")

    X = train_df[feature_cols]
    X_test = test_df[feature_cols]

    base_params = {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": 0.03,
        "num_leaves": 47,
        "min_child_samples": 80,
        "reg_lambda": 20.0,
        "max_delta_step": 1.5,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "feature_fraction": 0.7,
        "cat_smooth": 50.0,
        "min_data_per_group": 200,
        "max_cat_to_onehot": 4,
        "num_threads": -1,
        "verbose": -1,
    }

    if CV_MODE == "time":
        ts = train_df[TIME_COL]
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_localize(None)
        periods = ts.dt.to_period("M")

        month_counts = periods.value_counts()
        major_months = month_counts[month_counts >= 0.05 * len(train_df)].index
        val_period = major_months.max()

        tr_mask = periods < val_period
        val_mask = periods == val_period

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_va, y_va = X[val_mask], y[val_mask]

        print(f"Time split -> Train: {len(y_tr):,} rows ({ts[tr_mask].min()} to {ts[tr_mask].max()})")
        print(f"Time split -> Val:   {len(y_va):,} rows ({ts[val_mask].min()} to {ts[val_mask].max()})")
        assert len(y_tr) > 0 and len(y_va) > 0, "Empty train or validation split! Check timestamps."

        tr_set = lgb.Dataset(X_tr, label=y_tr)
        va_set = lgb.Dataset(X_va, label=y_va, reference=tr_set)

        val_probs = np.zeros((len(y_va), num_classes), dtype="float32")
        gain_total = np.zeros(len(feature_cols), dtype="float64")
        best_iters_list = []

        for idx, seed in enumerate(SEEDS, 1):
            print(f"\n--- Seed {idx}/{len(SEEDS)} (seed={seed}) ---")
            params = {**base_params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
            model = lgb.train(
                params,
                tr_set,
                num_boost_round=3000,
                valid_sets=[va_set],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
            )
            seed_probs = model.predict(X_va)
            val_probs += seed_probs / len(SEEDS)
            gain_total += model.feature_importance(importance_type="gain") / len(SEEDS)
            best_iters_list.append(max(50, int((model.best_iteration or 200) * 1.1)))
            seed_raw_f1 = f1_score(y_va, np.argmax(seed_probs, axis=1), average="macro")
            print(f"Seed {seed} best_iter={model.best_iteration} | Raw Macro-F1={seed_raw_f1:.5f}")
            del model
            gc.collect()

        del tr_set, va_set
        gc.collect()

        print("\nScoring 3-seed September holdout predictions...")
        raw_preds = np.argmax(val_probs, axis=1)
        print(f"Raw argmax Holdout Macro-F1: {f1_score(y_va, raw_preds, average='macro'):.5f}")

        # Split-half check inside September to verify multipliers generalize without overfitting
        half = len(y_va) // 2
        w_half = optimize_class_multipliers(val_probs[:half], y_va[:half])
        half2_raw_f1 = f1_score(y_va[half:], np.argmax(val_probs[half:], axis=1), average="macro")
        half2_opt_f1 = f1_score(y_va[half:], np.argmax(val_probs[half:] * w_half, axis=1), average="macro")
        print(f"Split-half multiplier check (2nd half of Sep): raw={half2_raw_f1:.5f} -> tuned={half2_opt_f1:.5f}")

        # Fit final 12 multipliers across all of September
        class_multipliers = optimize_class_multipliers(val_probs, y_va)
        opt_preds = np.argmax(val_probs * class_multipliers, axis=1)
        print(f"Tuned Holdout Macro-F1:      {f1_score(y_va, opt_preds, average='macro'):.5f}")
        print(
            classification_report(
                y_va,
                opt_preds,
                labels=np.arange(num_classes),
                target_names=le.classes_,
                digits=4,
                zero_division=0,
            )
        )
        print_top_confusions(y_va, opt_preds, le.classes_)

        imp = pd.Series(gain_total, index=feature_cols)
        print("\nTop 20 features (by mean gain across seeds):")
        print(imp.sort_values(ascending=False).head(20).round(1))

        # Refit 3-seed ensemble on all Jul-Sep data
        full_set = lgb.Dataset(X, label=y)
        test_probs = np.zeros((len(test_df), num_classes), dtype="float32")
        trained_models = []

        for idx, (seed, best_iters) in enumerate(zip(SEEDS, best_iters_list), 1):
            print(f"\nRetraining full seed {idx}/{len(SEEDS)} on all Jul-Sep rows ({best_iters} trees)...")
            params = {**base_params, "seed": seed, "bagging_seed": seed, "feature_fraction_seed": seed}
            full_model = lgb.train(
                params,
                full_set,
                num_boost_round=best_iters,
                callbacks=[lgb.log_evaluation(100)],
            )
            test_probs += full_model.predict(X_test) / len(SEEDS)
            trained_models.append(full_model)

    else:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEEDS[0])
        oof_probs = np.zeros((len(train_df), num_classes), dtype="float32")
        test_probs = np.zeros((len(test_df), num_classes), dtype="float32")
        trained_models = []

        for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
            X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
            X_va, y_va = X.iloc[va_idx], y[va_idx]

            tr_set = lgb.Dataset(X_tr, label=y_tr)
            va_set = lgb.Dataset(X_va, label=y_va, reference=tr_set)
            params = {**base_params, "seed": SEEDS[0] + fold}
            model = lgb.train(
                params,
                tr_set,
                num_boost_round=3000,
                valid_sets=[va_set],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(200)],
            )
            oof_probs[va_idx] = model.predict(X_va)
            test_probs += model.predict(X_test) / N_FOLDS
            trained_models.append(model)
            fold_f1 = f1_score(y_va, np.argmax(oof_probs[va_idx], axis=1), average="macro")
            print(f"Fold {fold} Raw Macro-F1: {fold_f1:.5f}")

        class_multipliers = optimize_class_multipliers(oof_probs, y)
        oof_preds = np.argmax(oof_probs * class_multipliers, axis=1)
        print(f"\nRaw OOF Macro-F1:   {f1_score(y, np.argmax(oof_probs, axis=1), average='macro'):.5f}")
        print(f"Tuned OOF Macro-F1: {f1_score(y, oof_preds, average='macro'):.5f}")
        print(
            classification_report(
                y,
                oof_preds,
                labels=np.arange(num_classes),
                target_names=le.classes_,
                digits=4,
                zero_division=0,
            )
        )

    print("\nLearned class probability multipliers:")
    for cls_name, mult in zip(le.classes_, class_multipliers):
        print(f"  {cls_name:22s}: {mult:.3f}")

    return test_probs, le, trained_models, class_multipliers


# ==============================================================================
# 5. EXPORT SUBMISSION
# ==============================================================================
def save_submission(
    test_df: pd.DataFrame,
    test_probs: np.ndarray,
    le: LabelEncoder,
    class_multipliers: np.ndarray,
) -> None:
    pred_labels = le.inverse_transform(np.argmax(test_probs * class_multipliers, axis=1))
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: pred_labels})

    # Restore original test.csv row order
    orig_ids = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=[ID_COL])
    sub = orig_ids.merge(sub, on=ID_COL, how="left")
    sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSaved {SUBMISSION_PATH} ({len(sub):,} rows)")
    print(sub[TARGET_COL].value_counts())


# ==============================================================================
# 6. EXPORT MODEL ARTIFACTS
# ==============================================================================
def save_model_artifacts(
    models: list[lgb.Booster],
    le: LabelEncoder,
    feature_cols: list[str],
    class_multipliers: np.ndarray,
    out_dir: str = MODEL_DIR,
) -> None:
    """Save trained LightGBM booster(s), encoder, multipliers, and feature list locally."""
    save_dir = Path(out_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for idx, booster in enumerate(models, 1):
        model_path = save_dir / (f"lgb_model_{idx}.txt" if len(models) > 1 else "lgb_model.txt")
        booster.save_model(str(model_path))

    with open(save_dir / "metadata.pkl", "wb") as f:
        pickle.dump(
            {
                "label_encoder": le,
                "feature_cols": feature_cols,
                "class_multipliers": class_multipliers,
            },
            f,
        )

    print(f"\nExported {len(models)} model file(s) + metadata.pkl to '{save_dir}/'")


if __name__ == "__main__":
    train_raw, test_raw = load_data()
    train_feat, test_feat, features = add_features(train_raw, test_raw)
    del train_raw, test_raw
    gc.collect()

    probs, encoder, trained_models, multipliers = train_and_predict(train_feat, test_feat, features)
    save_submission(test_feat, probs, encoder, multipliers)
    save_model_artifacts(trained_models, encoder, features, multipliers)
