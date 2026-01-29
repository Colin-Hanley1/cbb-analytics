#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path

OUTDIR = Path("mc_out")
OUT_JS = Path("mc_data.js")

FIELD_FILE = OUTDIR / "mc_field_odds.csv"
SEED_FILE_WIDE = OUTDIR / "mc_seed_odds_wide.csv"   # <-- NEW NAME

def _coerce_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def _load_seed_probs_wide(seed_path: Path) -> dict:
    """
    Expects: Team,1,2,...,16  (or Team,Seed_1,...,Seed_16)
    Returns: { Team: { "1": p, ..., "16": p } }
    """
    if not seed_path.exists():
        return {}

    sdf = pd.read_csv(seed_path)
    if "Team" not in sdf.columns:
        raise ValueError(f"{seed_path} missing required column 'Team'")

    sdf["Team"] = sdf["Team"].astype(str)

    # Detect which column naming scheme is present.
    # Priority: plain numeric "1".."16", else "Seed_1".."Seed_16"
    numeric_cols = [str(s) for s in range(1, 17)]
    seed_cols = [f"Seed_{s}" for s in range(1, 17)]

    has_numeric = all(c in sdf.columns for c in numeric_cols)
    has_seed = all(c in sdf.columns for c in seed_cols)

    if has_numeric:
        cols = numeric_cols
        sdf = _coerce_numeric(sdf, cols)
    elif has_seed:
        cols = seed_cols
        sdf = _coerce_numeric(sdf, cols)
    else:
        # If partially present, fill missing and proceed with whichever is more common
        present_numeric = sum(c in sdf.columns for c in numeric_cols)
        present_seed = sum(c in sdf.columns for c in seed_cols)

        if present_numeric >= present_seed and present_numeric > 0:
            cols = numeric_cols
            sdf = _coerce_numeric(sdf, cols)
        elif present_seed > 0:
            cols = seed_cols
            sdf = _coerce_numeric(sdf, cols)
        else:
            raise ValueError(
                f"{seed_path} must contain columns Team,1..16 (or Team,Seed_1..Seed_16)"
            )

    # Normalize rows so probabilities sum to 1 (if sum>0)
    row_sum = sdf[cols].sum(axis=1)
    mask = row_sum > 0
    sdf.loc[mask, cols] = sdf.loc[mask, cols].div(row_sum[mask], axis=0)

    seed_map = {}
    for _, r in sdf.iterrows():
        team = r["Team"]
        if cols[0].startswith("Seed_"):
            seed_map[team] = {str(s): float(r[f"Seed_{s}"]) for s in range(1, 17)}
        else:
            seed_map[team] = {str(s): float(r[str(s)]) for s in range(1, 17)}

    return seed_map

def main():
    if not FIELD_FILE.exists():
        raise FileNotFoundError(f"Missing: {FIELD_FILE}")

    df = pd.read_csv(FIELD_FILE)

    # Normalize numeric columns for field odds
    for c in ["MakeField_Prob", "AutoBid_Prob", "AtLarge_Prob", "AtLargeIfNoAQ_Prob", "NoAQ_Prob"]:
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # Only include teams that EVER make the tournament in sims
    df = df[df["MakeField_Prob"] > 0].copy()
    df["Team"] = df["Team"].astype(str)

    # Load seed distributions
    seed_map = _load_seed_probs_wide(SEED_FILE_WIDE)

    mc_map = {}
    for _, r in df.iterrows():
        team = r["Team"]
        mc_map[team] = {
            "field": float(r["MakeField_Prob"]),
            "aq": float(r["AutoBid_Prob"]),
            "atlarge": float(r["AtLarge_Prob"]),
            "noaq": float(r["NoAQ_Prob"]),
            "al_noaq": float(r["AtLargeIfNoAQ_Prob"]),
            # seeds default to zeros if not found
            "seeds": seed_map.get(team, {str(s): 0.0 for s in range(1, 17)}),
        }

    OUT_JS.write_text(
        "window.MC_DATA=" + json.dumps(mc_map, separators=(",", ":")) + ";\n",
        encoding="utf-8"
    )
    print(f"Wrote {OUT_JS.resolve()} with {len(mc_map)} teams.")
    if not SEED_FILE_WIDE.exists():
        print(f"Note: {SEED_FILE_WIDE} not found; wrote zeroed seed distributions.")

if __name__ == "__main__":
    main()
