"""
Compare the generated submission CSV against the hackathon sample submission
to verify formatting before uploading.

Run:
    python check_submission.py
"""

from pathlib import Path
import pandas as pd

# ==============================================================================
# CONFIGURE FILE PATHS HERE
# ==============================================================================
SUBMISSION_PATH = "./submissions.csv"
SAMPLE_SUBMISSION_PATH = "./sample_submissions.csv"


def check_submission(sub_path: str, sample_path: str) -> bool:
    sub_file = Path(sub_path)
    sample_file = Path(sample_path)

    if not sub_file.exists():
        raise FileNotFoundError(f"Submission file not found: {sub_file}")
    if not sample_file.exists():
        raise FileNotFoundError(f"Sample submission file not found: {sample_file}")

    sub = pd.read_csv(sub_file)
    sample = pd.read_csv(sample_file)

    passed = True

    def report(ok: bool, msg: str) -> None:
        nonlocal passed
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {msg}")
        if not ok:
            passed = False

    print(f"Comparing:\n  Submission: {sub_file}\n  Sample:     {sample_file}\n")

    # 1. Exact columns and column order
    cols_match = list(sub.columns) == list(sample.columns)
    report(
        cols_match,
        f"Columns match: {list(sub.columns)} == {list(sample.columns)}",
    )

    # 2. Exact row count
    rows_match = len(sub) == len(sample)
    report(
        rows_match,
        f"Row count matches: {len(sub):,} == {len(sample):,}",
    )

    # 3. No missing / NaN values
    null_counts = sub.isna().sum().to_dict()
    no_nulls = sum(null_counts.values()) == 0
    report(no_nulls, f"No NaN/missing values (null counts: {null_counts})")

    # 4. ID checks (assuming first column is ID and second is target label)
    if cols_match and len(sample.columns) >= 2:
        id_col = sample.columns[0]
        target_col = sample.columns[1]

        no_dup_ids = sub[id_col].nunique(dropna=False) == len(sub)
        report(no_dup_ids, f"No duplicate IDs in '{id_col}'")

        if rows_match:
            exact_order = (sub[id_col].astype(str).values == sample[id_col].astype(str).values).all()
            same_id_set = set(sub[id_col].astype(str)) == set(sample[id_col].astype(str))
            report(same_id_set, f"Exact same set of '{id_col}' values as sample")
            report(exact_order, f"'{id_col}' row order matches sample row-by-row")

        # 5. Target label sanity checks
        n_pred_classes = sub[target_col].nunique(dropna=True)
        sample_classes = set(sample[target_col].dropna().astype(str).unique())
        sub_classes = set(sub[target_col].dropna().astype(str).unique())

        print(f"\nTarget column '{target_col}':")
        print(f"  Submission dtype: {sub[target_col].dtype} | Sample dtype: {sample[target_col].dtype}")
        print(f"  Unique predicted labels ({n_pred_classes}): {sorted(sub_classes)}")

        if len(sample_classes) > 1:
            valid_labels = sub_classes.issubset(sample_classes)
            report(
                valid_labels,
                f"All predicted labels exist in sample label set ({sorted(sample_classes)})",
            )
        else:
            print(
                f"  [INFO] Sample only uses placeholder label {sorted(sample_classes)}; "
                "verify the 12 predicted label names above match the competition spec."
            )

        print("\nPredicted class distribution:")
        print(sub[target_col].value_counts())

    print("\nFirst 5 rows of submission:")
    print(sub.head())
    print("\nResult:", "READY TO SUBMIT" if passed else "FIX ISSUES ABOVE BEFORE SUBMITTING")
    return passed


if __name__ == "__main__":
    check_submission(SUBMISSION_PATH, SAMPLE_SUBMISSION_PATH)
