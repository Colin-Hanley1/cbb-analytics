#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

OUTDIR = Path("mc_out")
OUT_JS = Path("mc_data.js")

def main():
    field_path = OUTDIR / "mc_field_odds.csv"
    if not field_path.exists():
        raise FileNotFoundError(f"Missing: {field_path}")

    df = pd.read_csv(field_path)

    # Normalize numeric columns
    for c in ["MakeField_Prob", "AutoBid_Prob", "AtLarge_Prob", "AtLargeIfNoAQ_Prob", "NoAQ_Prob"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Only include teams that EVER make the tournament in sims
    df = df[df["MakeField_Prob"] > 0].copy()

    df["Team"] = df["Team"].astype(str)

    mc_map = {
        r["Team"]: {
            "field": float(r["MakeField_Prob"]),
            "aq": float(r["AutoBid_Prob"]),
            "atlarge": float(r["AtLarge_Prob"]),
            "noaq": float(r["NoAQ_Prob"]),                 # optional
            "al_noaq": float(r["AtLargeIfNoAQ_Prob"]),     # <-- THE ONE YOU WANT
        }
        for _, r in df.iterrows()
    }

    OUT_JS.write_text("window.MC_DATA=" + json.dumps(mc_map, separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"Wrote {OUT_JS.resolve()} with {len(mc_map)} teams.")

if __name__ == "__main__":
    main()
