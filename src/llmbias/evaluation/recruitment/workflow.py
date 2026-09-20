"""File inputs and output handling for the domain evaluation."""

def evaluate_recruitment(neutral_csv, sensitive_csv, model_name):
    import pandas as pd
    from llmbias.evaluation.recruitment.scoring import add_reference_columns, summarise
    rows = add_reference_columns(pd.read_csv(neutral_csv), pd.read_csv(sensitive_csv))
    combo = summarise(rows, ["gender", "ethnicity"], "gender_ethnicity")
    gender = summarise(rows, ["gender"], "gender")
    ethnic = summarise(rows, ["ethnicity"], "ethnicity")
    out = pd.concat([combo, gender, ethnic], ignore_index=True)
    out["model"] = model_name
    return out
