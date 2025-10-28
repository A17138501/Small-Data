#!/usr/bin/env python3
import argparse, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True)
ap.add_argument("--gold", required=True)
args = ap.parse_args()

pred = pd.read_csv(args.pred)
need = {"id","query","tools","raw_output","pred_tool","pred_args"}

if not need.issubset(pred.columns):
    gold = pd.read_csv(args.gold)[["query","tools"]]
    if "id" not in pred: pred["id"] = range(len(pred))
    if "query" not in pred: pred["query"] = gold["query"]
    if "tools" not in pred: pred["tools"] = gold["tools"]
    if "raw_output" not in pred: pred["raw_output"] = ""
    pred = pred[["id","query","tools","raw_output","pred_tool","pred_args"]]
    pred.to_csv(args.pred, index=False)
    print(f"✅ fixed {args.pred} rows={len(pred)}")
else:
    print(f"👍 columns OK: {args.pred}")
