"""Functions extracted from notebook cell 490 (zero-based); legacy globals may be required."""
import pandas as pd
from pathlib import Path
from llmbias.paths import legacy_root

REASONING_FILE = 'o4_mini_medbullets.csv'
STANDARD_MODELS = {'gpt-4.1': 'gpt4.1_medbullets_final.csv', 'gpt-4.1-mini': 'gpt4.1_mini_medbullets.csv', 'claude': 'claude_medbullets_cleaned.csv'}
DATA_DIR = legacy_root() / 'experiments/medical_data/output_files/medbullets'
CLAUDE_PRED_COL = 'clean_answer'
DEFAULT_PRED_COL = 'answer'
REASONING_NAME = "o4-mini"
GT_CANDIDATES = ["gt", "gold", "label", "correct_answer", "ground_truth", "gt_answer", "y", "target"]
def _normalize_base(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["case_id"] = df["case_id"].astype(str)
    df["ethnicity"] = df["ethnicity"].astype(str).str.strip().str.lower()
    df["gender"] = df["gender"].astype(str).str.strip().str.lower()
    df["variant"] = df["ethnicity"] + "_" + df["gender"]
    df["pred"] = df["pred"].astype(str).str.strip().str.upper()
    return df
def load_model(csv_path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # required columns
    required = {"index", "ethnicity", "gender"}
    if not required.issubset(df.columns):
        raise ValueError(f"[{model_name}] Missing required columns {required - set(df.columns)}. Found: {list(df.columns)}")

    if model_name == "claude":
        if CLAUDE_PRED_COL not in df.columns:
            raise ValueError(f"[claude] Expected '{CLAUDE_PRED_COL}' in columns. Found: {list(df.columns)}")
        pred_col = CLAUDE_PRED_COL
    else:
        if DEFAULT_PRED_COL not in df.columns:
            raise ValueError(f"[{model_name}] Expected '{DEFAULT_PRED_COL}' in columns. Found: {list(df.columns)}")
        pred_col = DEFAULT_PRED_COL

    out = df[["index", "ethnicity", "gender", pred_col]].copy()
    out = out.rename(columns={"index": "case_id", pred_col: "pred"})
    out["model"] = model_name
    out = _normalize_base(out)
    return out[["model", "case_id", "ethnicity", "gender", "variant", "pred"]]
def find_gt_column(df: pd.DataFrame) -> str:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in GT_CANDIDATES:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    raise ValueError(f"No GT column found. Looked for {GT_CANDIDATES}. Columns: {list(df.columns)}")
def load_gt_from_gpt41(csv_path: Path) -> pd.DataFrame:
    """
    Loads GT from GPT-4.1 CSV. GT must be present as a column (gt/gold/label/etc).
    Returns columns: case_id, variant, gt
    """
    df = pd.read_csv(csv_path)

    required = {"index", "ethnicity", "gender"}
    if not required.issubset(df.columns):
        raise ValueError(f"[GT:gpt-4.1] Missing required columns {required - set(df.columns)}. Found: {list(df.columns)}")

    gt_col = find_gt_column(df)

    gt = df[["index", "ethnicity", "gender", gt_col]].copy()
    gt = gt.rename(columns={"index": "case_id", gt_col: "gt"})
    gt["case_id"] = gt["case_id"].astype(str)
    gt["ethnicity"] = gt["ethnicity"].astype(str).str.strip().str.lower()
    gt["gender"] = gt["gender"].astype(str).str.strip().str.lower()
    gt["variant"] = gt["ethnicity"] + "_" + gt["gender"]
    gt["gt"] = gt["gt"].astype(str).str.strip().str.upper()

    # keep one GT per (case_id, variant)
    gt = gt.drop_duplicates(subset=["case_id", "variant"])
    return gt[["case_id", "variant", "gt"]]
def flip_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per (model, case_id): how many variants, how many unique preds, whether flips.
    """
    g = df.groupby(["model", "case_id"], as_index=False).agg(
        n_variants=("variant", "nunique"),
        n_unique=("pred", pd.Series.nunique),
        preds=("pred", lambda s: sorted(set(s.dropna().tolist()))),
    )
    g["flips"] = g["n_unique"] > 1
    return g
def main():
    # Load all predictions
    reasoning = load_model(DATA_DIR / REASONING_FILE, REASONING_NAME)

    standards = []
    for name, fn in STANDARD_MODELS.items():
        standards.append(load_model(DATA_DIR / fn, name))
    standards = pd.concat(standards, ignore_index=True)

    all_df = pd.concat([reasoning, standards], ignore_index=True)

    # Load GT from GPT-4.1 CSV
    gt_df = load_gt_from_gpt41(DATA_DIR / STANDARD_MODELS["gpt-4.1"])

    # Merge GT into all_df
    all_df = all_df.merge(gt_df, on=["case_id", "variant"], how="left")

    # Compute flip stats
    stats = flip_stats(all_df)

    # Reasoning stability per case_id
    r = stats[stats["model"] == REASONING_NAME].copy().rename(columns={
        "n_variants": "reasoning_n_variants",
        "n_unique": "reasoning_n_unique",
        "preds": "reasoning_preds",
        "flips": "reasoning_flips",
    })[["case_id", "reasoning_n_variants", "reasoning_n_unique", "reasoning_preds", "reasoning_flips"]]

    # Standard flip stats per case_id
    s = stats[stats["model"] != REASONING_NAME].copy().rename(columns={
        "n_variants": "standard_n_variants",
        "n_unique": "standard_n_unique",
        "preds": "standard_preds",
        "flips": "standard_flips",
    })

    merged = s.merge(r, on="case_id", how="inner")

    # Condition A: standard flips AND reasoning stable
    flipped = merged[(merged["standard_flips"]) & (~merged["reasoning_flips"])].copy()

    # Ensure both saw same number of variants (avoid missing data artifacts)
    flipped = flipped[flipped["standard_n_variants"] == flipped["reasoning_n_variants"]]

    # Save summary (Condition A)
    summary = flipped.sort_values(["model", "case_id"])[[
        "model", "case_id",
        "standard_n_variants", "standard_n_unique", "standard_preds",
        "reasoning_n_unique", "reasoning_preds"
    ]]
    summary_path = DATA_DIR / "flipped_cases_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path} (rows={len(summary)})")

    # Detailed per-variant view + filter to strong qualitative cases using GT:
    # Condition B (strong): reasoning correct for ALL variants; standard wrong for >=1 variant
    detailed_rows = []
    for _, row in summary.iterrows():
        case_id = row["case_id"]
        std_model = row["model"]

        sub = all_df[all_df["case_id"] == case_id].copy()

        # keep only rows with GT available
        sub = sub[sub["gt"].notna()]
        if sub.empty:
            continue

        pv = sub.pivot_table(index="variant", columns="model", values="pred", aggfunc="first").reset_index()

        # require both std model and reasoning present
        if REASONING_NAME not in pv.columns or std_model not in pv.columns:
            continue

        # attach GT per variant
        gt_map = sub.drop_duplicates("variant").set_index("variant")["gt"]
        pv["gt"] = pv["variant"].map(gt_map)

        # Strong condition: reasoning correct for all variants; standard wrong for at least one
        reasoning_all_correct = (pv[REASONING_NAME] == pv["gt"]).all()
        standard_any_wrong = (pv[std_model] != pv["gt"]).any()

        if not (reasoning_all_correct and standard_any_wrong):
            continue

        # keep nice columns
        pv = pv[["variant", "gt", REASONING_NAME, std_model]].copy()
        pv.insert(0, "standard_model", std_model)
        pv.insert(1, "case_id", case_id)
        detailed_rows.append(pv)

    if detailed_rows:
        detailed = pd.concat(detailed_rows, ignore_index=True)
        detailed_path = DATA_DIR / "flipped_cases_detailed.csv"
        detailed.to_csv(detailed_path, index=False)
        print(f"Saved: {detailed_path} (rows={len(detailed)})")
    else:
        print("No strong cases found after GT filtering (reasoning correct on all variants + standard wrong on >=1).")
