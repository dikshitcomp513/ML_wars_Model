"""
Hyperparameter Tuning on Locked `tuned2` Features (main9B_tune_hparams.py)
1. Locks the exact `main9B_tuned2.py` causal feature set (built once).
2. Evaluates 7 targeted LightGBM configurations on the Jul-Aug -> Sep split
   using fast single-seed runs (lr=0.05, early_stopping=60) ranked by
   out-of-sample 2nd-half Sep Split-Half Tuned Macro-F1.
3. Automatically takes the winning hyperparameters, trains a 2-seed ensemble at
   lr=0.03 on Jul-Aug -> Sep, retrains on all Jul-Sep rows, and saves
   submission_9B_tuned3.csv + ./saved_model_9B_tuned3/.

Run:
    python main9B_tune_hparams.py
"""

import gc
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, f1_score, log_loss
from sklearn.preprocessing import LabelEncoder

# ==============================================================================
# 1. SETTINGS
# ==============================================================================
DATA_DIR = "./data"
ID_COL = "txn_id"
TARGET_COL = "label"
TIME_COL = "timestamp"
AMOUNT_COL = "amount"
ACC_COL = "sender_id"
MERCH_COL = "receiver_id"

SEED = 42
SEARCH_LR = 0.06
SEARCH_EARLY_STOP = 50
SEARCH_MAX_BIN = 127
FINAL_LR = 0.03
FINAL_EARLY_STOP = 90
FINAL_MAX_BIN = 255
FINAL_SEEDS = (42, 1024)
RUN_FINAL_RETRAIN = True

MODEL_DIR = "./saved_model_9B_tuned3"
SUBMISSION_PATH = "submission_9B_tuned3.csv"

# 5 focused, high-ROI hyperparameter candidates around the locked `tuned2` features
TRIAL_CONFIGS = [
    {
        "name": "01_baseline_tuned2",
        "num_leaves": 47,
        "min_child_samples": 80,
        "reg_lambda": 20.0,
        "reg_alpha": 0.0,
        "max_delta_step": 1.5,
        "feature_fraction": 0.70,
        "bagging_fraction": 0.80,
        "cat_smooth": 50.0,
        "min_data_per_group": 200,
        "extra_trees": False,
        "weight_power": 0.0,
    },
    {
        "name": "02_leaves31_tight_cat20",
        "num_leaves": 31,
        "min_child_samples": 60,
        "reg_lambda": 35.0,
        "reg_alpha": 2.0,
        "max_delta_step": 1.2,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.85,
        "cat_smooth": 20.0,
        "min_data_per_group": 150,
        "extra_trees": False,
        "weight_power": 0.0,
    },
    {
        "name": "03_leaves47_ff80_cat15",
        "num_leaves": 47,
        "min_child_samples": 80,
        "reg_lambda": 25.0,
        "reg_alpha": 1.0,
        "max_delta_step": 1.5,
        "feature_fraction": 0.80,
        "bagging_fraction": 0.85,
        "cat_smooth": 15.0,
        "min_data_per_group": 100,
        "extra_trees": False,
        "weight_power": 0.0,
    },
    {
        "name": "04_extra_trees_47",
        "num_leaves": 47,
        "min_child_samples": 80,
        "reg_lambda": 20.0,
        "reg_alpha": 1.0,
        "max_delta_step": 1.5,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.80,
        "cat_smooth": 30.0,
        "min_data_per_group": 150,
        "extra_trees": True,
        "weight_power": 0.0,
    },
    {
        "name": "05_gentle_weight_015",
        "num_leaves": 47,
        "min_child_samples": 80,
        "reg_lambda": 20.0,
        "reg_alpha": 1.0,
        "max_delta_step": 1.5,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.80,
        "cat_smooth": 30.0,
        "min_data_per_group": 150,
        "extra_trees": False,
        "weight_power": 0.15,
    },
]


# ==============================================================================
# 2. LOAD & JOIN DATA
# ==============================================================================
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Loading data...")
    train = pd.read_csv(f"{DATA_DIR}/train.csv", parse_dates=[TIME_COL])
    test = pd.read_csv(f"{DATA_DIR}/test.csv", parse_dates=[TIME_COL])
    raw_accounts = pd.read_csv(f"{DATA_DIR}/accounts.csv").drop_duplicates(subset="account_id")
    merchants = pd.read_csv(f"{DATA_DIR}/merchants.csv").drop_duplicates(subset="merchant_id")

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
# 3. LOCKED `tuned2` VECTORIZED FEATURES
# ==============================================================================
def add_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    print("Building locked tuned2 features...")
    train = train.copy()
    test = test.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test[TARGET_COL] = np.nan

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

    # --- P2P receiver profile & sender-receiver relationship flags ---
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

    # --- Account velocity, spend & behavioral baselines ---
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

    min5_bucket = df[TIME_COL].dt.floor("5min")
    hour_bucket = df[TIME_COL].dt.floor("1h")
    day_bucket = df[TIME_COL].dt.floor("1D")

    df["acc_tx_in_5min"] = df.groupby([ACC_COL, min5_bucket], sort=False).cumcount().astype("int32")
    df["acc_tx_in_hour"] = df.groupby([ACC_COL, hour_bucket], sort=False).cumcount().astype("int32")
    df["acc_tx_in_day"] = df.groupby([ACC_COL, day_bucket], sort=False).cumcount().astype("int32")

    g_acc_day = df.groupby([ACC_COL, day_bucket], sort=False)
    df["acc_amt_in_day"] = (g_acc_day[AMOUNT_COL].cumsum() - df[AMOUNT_COL]).astype("float32")

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

    df["merch_tx_per_sender_in_day"] = (
        df["merch_tx_in_day"] / (df["recv_unique_senders_in_day"] + 1.0)
    ).astype("float32")

    # Sender inflow-to-outflow layering signals (cross-midnight matching)
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

        df["acc_device_prior_cnt"] = df.groupby([ACC_COL, "device_id"], sort=False).cumcount().astype("int32")
        df["is_new_device_for_sender"] = (df["acc_device_prior_cnt"] == 0).astype("int8")

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
# 4. TUNING & FINAL ENSEMBLE
# ==============================================================================
def get_sample_weights(y: np.ndarray, power: float) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = {c: (counts.max() / n) ** power for c, n in zip(classes, counts)}
    return np.array([weights[val] for val in y], dtype="float32")


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
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
    n_grid: int = 65,
) -> np.ndarray:
    num_classes = probs.shape[1]
    counts = np.bincount(y_true, minlength=num_classes)
    prior_init = (counts.max() / np.maximum(counts, 1)) ** 0.35
    prior_init /= prior_init.mean()

    grid = np.geomspace(0.005, 100.0, n_grid)
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


def cfg_to_lgb_params(cfg: dict, num_classes: int, lr: float, seed: int, max_bin: int = 255) -> dict:
    return {
        "objective": "multiclass",
        "num_class": num_classes,
        "metric": "multi_logloss",
        "learning_rate": lr,
        "max_bin": max_bin,
        "feature_pre_filter": False,
        "num_leaves": cfg["num_leaves"],
        "min_child_samples": cfg["min_child_samples"],
        "reg_lambda": cfg["reg_lambda"],
        "reg_alpha": cfg["reg_alpha"],
        "max_delta_step": cfg["max_delta_step"],
        "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"],
        "bagging_freq": 1,
        "cat_smooth": cfg["cat_smooth"],
        "min_data_per_group": cfg["min_data_per_group"],
        "max_cat_to_onehot": 4,
        "extra_trees": cfg["extra_trees"],
        "num_threads": -1,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "verbose": -1,
    }


def run_hparam_tuning(
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

    # Build base Datasets ONCE with feature_pre_filter=False & max_bin=127 for fast, safe reuse
    ds_params_search = {"feature_pre_filter": False, "max_bin": SEARCH_MAX_BIN}
    tr_base = lgb.Dataset(X_tr, label=y_tr, params=ds_params_search, free_raw_data=False)
    va_base = lgb.Dataset(X_va, label=y_va, reference=tr_base, params=ds_params_search, free_raw_data=False)

    half = len(y_va) // 2
    results = []

    print(f"\nRunning {len(TRIAL_CONFIGS)} hyperparameter trials (lr={SEARCH_LR}, max_bin={SEARCH_MAX_BIN}, seed={SEED})...")
    for idx, cfg in enumerate(TRIAL_CONFIGS, 1):
        t0 = time.perf_counter()
        wp = cfg["weight_power"]
        tr_set = (
            tr_base
            if wp == 0.0
            else lgb.Dataset(
                X_tr,
                label=y_tr,
                weight=get_sample_weights(y_tr, wp),
                reference=tr_base,
                params=ds_params_search,
                free_raw_data=False,
            )
        )
        params = cfg_to_lgb_params(cfg, num_classes, SEARCH_LR, SEED, max_bin=SEARCH_MAX_BIN)
        model = lgb.train(
            params,
            tr_set,
            num_boost_round=2500,
            valid_sets=[va_base],
            callbacks=[lgb.early_stopping(SEARCH_EARLY_STOP, verbose=False)],
        )

        val_probs = model.predict(X_va)
        val_ll = log_loss(y_va, val_probs, labels=np.arange(num_classes))
        raw_f1 = fast_macro_f1(y_va, np.argmax(val_probs, axis=1), num_classes)

        w_h1 = optimize_class_multipliers(val_probs[:half], y_va[:half], n_passes=4, n_grid=40)
        h2_tuned = fast_macro_f1(y_va[half:], np.argmax(val_probs[half:] * w_h1, axis=1), num_classes)

        w_all = optimize_class_multipliers(val_probs, y_va, n_passes=4, n_grid=40)
        full_tuned = fast_macro_f1(y_va, np.argmax(val_probs * w_all, axis=1), num_classes)

        dt = time.perf_counter() - t0
        rec = {
            "cfg": cfg,
            "name": cfg["name"],
            "best_iter": int(model.best_iteration),
            "logloss": float(val_ll),
            "raw_f1": float(raw_f1),
            "split_half_f1": float(h2_tuned),
            "full_sep_f1": float(full_tuned),
            "sec": float(dt),
        }
        results.append(rec)
        print(
            f"[{idx}/{len(TRIAL_CONFIGS)}] {cfg['name']:24s} | iter={rec['best_iter']:3d} | "
            f"logloss={rec['logloss']:.5f} | raw={rec['raw_f1']:.5f} | "
            f"split_half={rec['split_half_f1']:.5f} | full_sep={rec['full_sep_f1']:.5f} ({dt:.0f}s)"
        )
        del model
        gc.collect()

    del tr_base, va_base
    gc.collect()

    # Rank by out-of-sample Split-Half Macro-F1 (fair across weighted & unweighted)
    results.sort(key=lambda r: (r["split_half_f1"], r["full_sep_f1"]), reverse=True)
    print("\n=== HYPERPARAMETER TUNING LEADERBOARD (sorted by 2nd-half Sep OOS F1) ===")
    for rank, r in enumerate(results, 1):
        print(
            f"  #{rank} {r['name']:24s} | iter={r['best_iter']:3d} | logloss={r['logloss']:.5f} | "
            f"raw={r['raw_f1']:.5f} | split_half={r['split_half_f1']:.5f} | full_sep={r['full_sep_f1']:.5f}"
        )

    best_cfg = results[0]["cfg"]
    print(f"\nWinner: {best_cfg['name']} -> {best_cfg}")

    if not RUN_FINAL_RETRAIN:
        return np.zeros((len(test_df), num_classes), dtype="float32"), le, [], np.ones(num_classes, dtype="float32")

    # Train final multi-seed ensemble with the winning config at FINAL_LR (0.03) and full bins
    print(f"\nTraining final {len(FINAL_SEEDS)}-seed ensemble with winner '{best_cfg['name']}' (lr={FINAL_LR})...")
    wp = best_cfg["weight_power"]
    ds_params_final = {"feature_pre_filter": False, "max_bin": FINAL_MAX_BIN}
    tr_final = lgb.Dataset(
        X_tr,
        label=y_tr,
        weight=get_sample_weights(y_tr, wp) if wp > 0 else None,
        params=ds_params_final,
    )
    va_final = lgb.Dataset(X_va, label=y_va, reference=tr_final, params=ds_params_final)

    val_probs = np.zeros((len(y_va), num_classes), dtype="float32")
    best_iters_list = []

    for idx, s in enumerate(FINAL_SEEDS, 1):
        print(f"\n--- Final Seed {idx}/{len(FINAL_SEEDS)} (seed={s}) ---")
        params = cfg_to_lgb_params(best_cfg, num_classes, FINAL_LR, s, max_bin=FINAL_MAX_BIN)
        model = lgb.train(
            params,
            tr_final,
            num_boost_round=3000,
            valid_sets=[va_final],
            callbacks=[lgb.early_stopping(FINAL_EARLY_STOP), lgb.log_evaluation(100)],
        )
        seed_probs = model.predict(X_va)
        val_probs += seed_probs / len(FINAL_SEEDS)
        best_iters_list.append(max(50, int((model.best_iteration or 150) * 1.1)))
        del model
        gc.collect()

    del tr_final, va_final
    gc.collect()

    raw_preds = np.argmax(val_probs, axis=1)
    print(f"\nWinner Ensemble Raw Holdout Macro-F1: {f1_score(y_va, raw_preds, average='macro'):.5f}")
    w_half = optimize_class_multipliers(val_probs[:half], y_va[:half])
    h2_raw = f1_score(y_va[half:], np.argmax(val_probs[half:], axis=1), average="macro")
    h2_opt = f1_score(y_va[half:], np.argmax(val_probs[half:] * w_half, axis=1), average="macro")
    print(f"Winner Split-half check (2nd half Sep): raw={h2_raw:.5f} -> tuned={h2_opt:.5f}")

    class_multipliers = optimize_class_multipliers(val_probs, y_va)
    opt_preds = np.argmax(val_probs * class_multipliers, axis=1)
    print(f"Winner Tuned Holdout Macro-F1:      {f1_score(y_va, opt_preds, average='macro'):.5f}")
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

    # Retrain on all Jul-Sep rows for October test scoring
    full_set = lgb.Dataset(
        X,
        label=y,
        weight=get_sample_weights(y, wp) if wp > 0 else None,
        params=ds_params_final,
    )
    test_probs = np.zeros((len(test_df), num_classes), dtype="float32")
    trained_models = []

    for idx, (s, best_iters) in enumerate(zip(FINAL_SEEDS, best_iters_list), 1):
        print(f"\nRetraining winner seed {idx}/{len(FINAL_SEEDS)} on all Jul-Sep rows ({best_iters} trees)...")
        params = cfg_to_lgb_params(best_cfg, num_classes, FINAL_LR, s, max_bin=FINAL_MAX_BIN)
        full_model = lgb.train(
            params,
            full_set,
            num_boost_round=best_iters,
            callbacks=[lgb.log_evaluation(100)],
        )
        test_probs += full_model.predict(X_test) / len(FINAL_SEEDS)
        trained_models.append(full_model)

    print("\nLearned class probability multipliers:")
    for cls_name, mult in zip(le.classes_, class_multipliers):
        print(f"  {cls_name:22s}: {mult:.3f}")

    return test_probs, le, trained_models, class_multipliers


# ==============================================================================
# 5. EXPORT SUBMISSION & MODEL ARTIFACTS
# ==============================================================================
def save_submission(
    test_df: pd.DataFrame,
    test_probs: np.ndarray,
    le: LabelEncoder,
    class_multipliers: np.ndarray,
) -> None:
    pred_labels = le.inverse_transform(np.argmax(test_probs * class_multipliers, axis=1))
    sub = pd.DataFrame({ID_COL: test_df[ID_COL].values, TARGET_COL: pred_labels})

    orig_ids = pd.read_csv(f"{DATA_DIR}/test.csv", usecols=[ID_COL])
    sub = orig_ids.merge(sub, on=ID_COL, how="left")
    sub.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSaved {SUBMISSION_PATH} ({len(sub):,} rows)")
    print(sub[TARGET_COL].value_counts())


def save_model_artifacts(
    models: list[lgb.Booster],
    le: LabelEncoder,
    feature_cols: list[str],
    class_multipliers: np.ndarray,
    out_dir: str = MODEL_DIR,
) -> None:
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

    probs, encoder, trained_models, multipliers = run_hparam_tuning(train_feat, test_feat, features)
    if RUN_FINAL_RETRAIN and trained_models:
        save_submission(test_feat, probs, encoder, multipliers)
        save_model_artifacts(trained_models, encoder, features, multipliers)
