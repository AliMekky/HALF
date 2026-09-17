# save as merge_summaries.py
import json, pathlib, re, argparse, copy

def clean_summary(obj):
    return obj["response"]["body"]["choices"][0]["message"]["content"].strip()

def run(src_inputs, vendor_outputs, out_jsonl):
    # 1) load input variants -----------------------------------------------
    by_cid = {}
    for j in map(json.loads, open(src_inputs)):
        print(j.keys())
        by_cid[j["custom_id"]] = j

    # 2) merge in summaries -------------------------------------------------
    for row in map(json.loads, open(vendor_outputs)):
        cid   = row["custom_id"]
        summ  = clean_summary(row)
        inst  = by_cid[cid]
        inst["summary"] = summ
        yield inst

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs")
    p.add_argument("--outputs")
    p.add_argument("--out")
    a = p.parse_args()

    with open(a.out, "w") as w:
        for rec in run(a.inputs, a.outputs, a.out):
            w.write(json.dumps(rec) + "\n")
