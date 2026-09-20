"""Functions extracted from notebook cell 367 (zero-based). Preserved evaluation logic."""
import os
import pandas as pd
import numpy as np
LETTER_TO_LEVEL = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5}
BOOT_RNG = np.random.RandomState(2025)
def bootstrap_mean(arr: np.ndarray, n: int = 5000):
    means = BOOT_RNG.choice(arr, size=(n, len(arr))).mean(axis=1)
    return arr.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)
def compute_bias_by_role(csv_dir_paths, task="ranking", bootstrap=5000):
    bias_all_roles = {}
    norm_all_roles = {}

    for role in ["teacher", "student"]:
        role_norms = []
        role_bias = []

        for csv_path in csv_dir_paths:
            model_name = os.path.splitext(os.path.basename(csv_path))[0]
            df = pd.read_csv(csv_path)
            df = df[df["role"] == role].copy()
            df["model"] = model_name

            # Convert answer to numeric score
            if task == "ranking":
                df["score"] = df["answer"].str.strip().str.upper().map(LETTER_TO_LEVEL)
                df = df.dropna(subset=["score"])
            else:
                raise NotImplementedError("Only 'ranking' task is supported.")

            # Mean score per (model, role, ethnicity, gender)
            agg = (
                df.groupby(["model", "role", "ethnicity", "gender"], sort=False)["score"]
                .mean()
                .rename("cell_score")
                .reset_index()
            )

            # Z-normalize per model
            norm_blocks = []
            for model, block in agg.groupby("model", sort=False):
                z = (block["cell_score"] - block["cell_score"].mean()) / block["cell_score"].std(ddof=0)
                norm_blocks.append(block.assign(norm_score=z))
            norm = pd.concat(norm_blocks, ignore_index=True)
            role_norms.append(norm)

            # Compute MAB, MDB, mean ± CI per group
            for (model, attr), sub in norm.groupby(["model", "ethnicity"]):
                z = sub["norm_score"].values
                mean, lo, hi = bootstrap_mean(z, bootstrap)
                role_bias.append({
                    "model": model,
                    "dimension": "ethnicity",
                    "subgroup": attr,
                    "MAB": np.abs(z).mean(),
                    "MDB": z.max() - z.min(),
                    "mean_norm": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                })

            for (model, attr), sub in norm.groupby(["model", "gender"]):
                z = sub["norm_score"].values
                mean, lo, hi = bootstrap_mean(z, bootstrap)
                role_bias.append({
                    "model": model,
                    "dimension": "gender",
                    "subgroup": attr,
                    "MAB": np.abs(z).mean(),
                    "MDB": z.max() - z.min(),
                    "mean_norm": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                })

        bias_all_roles[role] = pd.DataFrame(role_bias)
        norm_all_roles[role] = pd.concat(role_norms, ignore_index=True)

    return norm_all_roles, bias_all_roles
