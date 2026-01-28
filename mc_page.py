#!/usr/bin/env python3
"""
mc_report.py

Read Monte Carlo outputs in mc_out/ and render an easy-to-digest HTML report.

Expected files in --outdir (default: mc_out):
- mc_field_odds.csv
- mc_seed_odds_wide.csv
- mc_seed_odds_long.csv

Optional (season-by-season logs):
- Any *.jsonl in outdir (e.g., mc_seasons.jsonl, seasons.jsonl, mc_seasons_log.jsonl)
  Each line should be a JSON object for one simulated season. This script will:
    - Count seasons
    - Summarize AQs / At-large lists if present
    - Show a sample of seasons
    - If the season object includes `field` (list of teams/seed/bid_type), it will summarize

Usage:
  python mc_report.py --outdir mc_out --html mc_report.html
"""

from __future__ import annotations

import os
import json
import math
import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ----------------------------
# Helpers
# ----------------------------
def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
        return df
    except Exception as e:
        print(f"[WARN] Failed to read {path.name}: {e}")
        return None


def find_jsonl_logs(outdir: Path) -> List[Path]:
    preferred = [
        outdir / "mc_seasons.jsonl",
        outdir / "seasons.jsonl",
        outdir / "mc_season_logs.jsonl",
        outdir / "mc_sim_seasons.jsonl",
    ]
    found = [p for p in preferred if p.exists()]
    if found:
        return found
    return sorted(outdir.glob("*.jsonl"))


def safe_float(x, fb=0.0) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else fb
    except Exception:
        return fb


def to_iso_now() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def clamp_int(x, lo, hi):
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except Exception:
        return lo


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Team" not in out.columns:
        for alt in ["team", "TEAM"]:
            if alt in out.columns:
                out = out.rename(columns={alt: "Team"})
                break
    return out


# ----------------------------
# Season log parsing (optional)
# ----------------------------
def load_jsonl(path: Path, max_lines: Optional[int] = None) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def summarize_seasons(seasons: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(seasons)
    if n == 0:
        return {"n": 0}

    aq_counter: Dict[str, int] = {}
    atlarge_counter: Dict[str, int] = {}
    seed_counter: Dict[str, Dict[int, int]] = {}
    made_field_counter: Dict[str, int] = {}

    def inc(d: Dict[str, int], k: str, by: int = 1):
        d[k] = d.get(k, 0) + by

    for s in seasons:
        aq_list = s.get("aq_list") or s.get("aqs") or s.get("AQs") or []
        if isinstance(aq_list, dict):
            aq_list = list(aq_list.values())
        if isinstance(aq_list, str):
            aq_list = [aq_list]
        if isinstance(aq_list, list):
            for t in aq_list:
                if t is None:
                    continue
                inc(aq_counter, str(t), 1)

        field = s.get("field") or s.get("field_df") or s.get("selected_field") or None
        if isinstance(field, list):
            for row in field:
                if isinstance(row, dict):
                    team = row.get("Team") or row.get("team")
                    if team is None:
                        continue
                    team = str(team)
                    inc(made_field_counter, team, 1)

                    bt = row.get("Bid_Type") or row.get("bid_type") or row.get("Bid") or None
                    if bt is not None:
                        bt_key = str(bt).lower().replace("-", "").replace(" ", "")
                        if bt_key in ["atlarge", "atlargebid", "at_large"]:
                            inc(atlarge_counter, team, 1)

                    seed = row.get("Seed") or row.get("seed")
                    if seed is not None:
                        seed = clamp_int(seed, 1, 16)
                        if team not in seed_counter:
                            seed_counter[team] = {}
                        seed_counter[team][seed] = seed_counter[team].get(seed, 0) + 1

        al_list = s.get("at_large") or s.get("atlarge") or s.get("at_large_list") or s.get("atlarge_list") or []
        if isinstance(al_list, dict):
            al_list = list(al_list.values())
        if isinstance(al_list, str):
            al_list = [al_list]
        if isinstance(al_list, list):
            for t in al_list:
                if t is None:
                    continue
                inc(atlarge_counter, str(t), 1)

    aq_df = pd.DataFrame([{"Team": k, "Count": v, "Prob": v / n} for k, v in aq_counter.items()])
    aq_df = aq_df.sort_values(["Prob", "Team"], ascending=[False, True]) if not aq_df.empty else aq_df

    al_df = pd.DataFrame([{"Team": k, "Count": v, "Prob": v / n} for k, v in atlarge_counter.items()])
    al_df = al_df.sort_values(["Prob", "Team"], ascending=[False, True]) if not al_df.empty else al_df

    mf_df = pd.DataFrame([{"Team": k, "Count": v, "Prob": v / n} for k, v in made_field_counter.items()])
    mf_df = mf_df.sort_values(["Prob", "Team"], ascending=[False, True]) if not mf_df.empty else mf_df

    seed_rows = []
    for team, d in seed_counter.items():
        for seed, c in d.items():
            seed_rows.append({"Team": team, "Seed": int(seed), "Count": int(c), "Prob": c / n})
    seed_df = pd.DataFrame(seed_rows)
    if not seed_df.empty:
        seed_df = seed_df.sort_values(["Team", "Seed"], ascending=[True, True])

    return {"n": n, "aq_df": aq_df, "al_df": al_df, "mf_df": mf_df, "seed_df": seed_df}


# ----------------------------
# HTML builder
# ----------------------------
def df_to_js_records(df: Optional[pd.DataFrame], max_rows: Optional[int] = None) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    if max_rows is not None:
        df = df.head(max_rows)
    out = df.replace({np.nan: None}).to_dict(orient="records")
    return out


def build_html(
    outdir: Path,
    df_field: Optional[pd.DataFrame],
    df_seed_wide: Optional[pd.DataFrame],
    df_seed_long: Optional[pd.DataFrame],
    season_logs: List[Tuple[str, List[Dict[str, Any]]]],
    season_summary: Dict[str, Any],
    title: str = "MC Report | CHan Analytics",
) -> str:
    now = to_iso_now()

    if df_field is not None:
        df_field = normalize_cols(df_field)
    if df_seed_wide is not None:
        df_seed_wide = normalize_cols(df_seed_wide)
    if df_seed_long is not None:
        df_seed_long = normalize_cols(df_seed_long)

    # Build "overview" table = field odds + seed mode + best seed info
    top_field = df_field.copy() if df_field is not None else None
    if top_field is not None and not top_field.empty:
        for c in ["MakeField_Prob", "AtLarge_Prob", "AutoBid_Prob", "AtLargeIfNoAQ_Prob", "NoAQ_Prob"]:
            if c in top_field.columns:
                top_field[c] = pd.to_numeric(top_field[c], errors="coerce").fillna(0.0)
        if "MakeField_Prob" in top_field.columns:
            top_field = top_field.sort_values("MakeField_Prob", ascending=False)

    seed_mode_df = None
    if df_seed_long is not None and not df_seed_long.empty and {"Team", "Seed", "Prob"}.issubset(df_seed_long.columns):
        tmp = df_seed_long.copy()
        tmp["Prob"] = pd.to_numeric(tmp["Prob"], errors="coerce").fillna(0.0)
        idx = tmp.sort_values(["Team", "Prob"], ascending=[True, False]).groupby("Team", as_index=False).head(1)
        seed_mode_df = idx.rename(columns={"Prob": "ModeSeed_Prob", "Seed": "ModeSeed"})[["Team", "ModeSeed", "ModeSeed_Prob"]]

    combined = None
    if top_field is not None and not top_field.empty:
        combined = top_field.copy()
        if seed_mode_df is not None and not seed_mode_df.empty:
            combined = combined.merge(seed_mode_df, on="Team", how="left")

        if df_seed_wide is not None and not df_seed_wide.empty and "Team" in df_seed_wide.columns:
            sw = df_seed_wide.copy()
            # Normalize seed columns to strings "1".."16"
            renamed = {}
            for c in sw.columns:
                if str(c).isdigit():
                    renamed[c] = str(int(float(c)))
            sw = sw.rename(columns=renamed)

            seed_cols = [c for c in sw.columns if str(c).isdigit()]
            for c in seed_cols:
                sw[c] = pd.to_numeric(sw[c], errors="coerce").fillna(0.0)

            def best_seed_row(r):
                best = None
                bestp = -1.0
                for s in range(1, 17):
                    cs = str(s)
                    if cs not in r:
                        continue
                    p = float(r[cs])
                    if p > bestp + 1e-12 or (abs(p - bestp) <= 1e-12 and (best is None or s < best)):
                        bestp = p
                        best = s
                return pd.Series({"BestSeed": best if best is not None else None, "BestSeed_Prob": bestp if bestp >= 0 else None})

            bs = sw.apply(best_seed_row, axis=1)
            sw2 = pd.concat([sw[["Team"]], bs], axis=1)
            combined = combined.merge(sw2, on="Team", how="left")

        # Choose display columns (only if they exist)
        nice_cols = ["Team"]
        for c in ["MakeField_Prob", "AtLarge_Prob", "AutoBid_Prob", "NoAQ_Prob", "AtLargeIfNoAQ_Prob"]:
            if c in combined.columns:
                nice_cols.append(c)
        for c in ["BestSeed", "BestSeed_Prob", "ModeSeed", "ModeSeed_Prob"]:
            if c in combined.columns:
                nice_cols.append(c)
        combined = combined[nice_cols].copy()

    # Samples
    season_samples = []
    for name, seasons in season_logs:
        for i, s in enumerate(seasons[:25]):
            season_samples.append({"source": name, "i": i, "json": s})

    # JS payloads
    js_field = df_to_js_records(df_field)
    js_overview = df_to_js_records(combined)
    js_seed_long = df_to_js_records(df_seed_long, max_rows=40000)

    js_aq = df_to_js_records(season_summary.get("aq_df"))
    js_al = df_to_js_records(season_summary.get("al_df"))

    # HTML (tables match your styling + sortable headers)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <link rel="shortcut icon" type="image/x-icon" href="favicon.ico" />
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>{title}</title>

  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">

  <script>
    tailwind.config = {{
      theme: {{
        extend: {{
          fontFamily: {{
            sans: ['Inter', 'sans-serif'],
            mono: ['JetBrains Mono', 'monospace'],
          }},
          colors: {{
            brand: {{ 500: '#6366f1', 600: '#4f46e5', 900: '#312e81' }}
          }}
        }}
      }}
    }}
  </script>

  <style>
    body {{ -webkit-font-smoothing: antialiased; }}
    ::-webkit-scrollbar {{ width: 8px; height: 8px; }}
    ::-webkit-scrollbar-track {{ background: #f1f5f9; }}
    ::-webkit-scrollbar-thumb {{ background: #cbd5e1; border-radius: 4px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: #94a3b8; }}
    #mobile-menu {{ display: none; transition: all 0.3s; }}
    #mobile-menu.open {{ display: block; }}
    html {{ scroll-behavior: smooth; }}
  </style>
</head>

<body class="bg-slate-50 text-slate-900 min-h-screen flex flex-col selection:bg-brand-500 selection:text-white">

  <!-- Navbar (same style language as your pages; you can swap links as desired) -->
  <nav class="bg-slate-900/95 backdrop-blur-md text-white shadow-lg sticky top-0 z-50 border-b border-slate-700/50" id="siteNav">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <div class="flex items-center gap-3">
          <a href="#topAnchor" onclick="scrollToTop(event)" class="flex items-center">
            <img src="wlogo.png" alt="CHan Analytics"
                 class="h-6 sm:h-7 w-auto object-contain select-none" style="max-height: 28px" />
          </a>

          <div class="hidden md:flex ml-8 space-x-1">
            <a href="index.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Rankings</a>
            <a href="schedule.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Today's Games</a>
            <a href="bracketology.html" class="text-slate-300 hover:text-white hover:bg-slate-800 px-3 py-2 rounded-md text-sm font-medium transition-colors">Bracketology</a>
            <a href="mc_report.html" class="bg-slate-800 text-white px-3 py-2 rounded-md text-sm font-medium transition-colors border border-slate-700">MC Report</a>
          </div>
        </div>

        <div class="flex items-center gap-4">
          <div class="hidden md:block text-xs font-mono text-slate-400 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
            <span class="text-brand-500">●</span> Updated: <span class="text-slate-200">{now}</span>
          </div>
          <div class="-mr-2 flex md:hidden">
            <button type="button" onclick="toggleMenu()" class="text-slate-400 hover:text-white p-2" aria-label="Open menu">
              <svg class="h-6 w-6" stroke="currentColor" fill="none" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <div class="md:hidden" id="mobile-menu">
      <div class="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-slate-900 border-t border-slate-800">
        <a href="index.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Rankings</a>
        <a href="schedule.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Today's Games</a>
        <a href="bracketology.html" class="text-slate-300 hover:text-white block px-3 py-2 rounded-md text-base font-medium">Bracketology</a>
        <a href="mc_report.html" class="bg-slate-800 text-white block px-3 py-2 rounded-md text-base font-medium border border-slate-700">MC Report</a>
      </div>
    </div>
  </nav>

  <main class="flex-grow max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 w-full">
    <div id="topAnchor" style="scroll-margin-top: 88px;"></div>

    <div class="mb-8">
      <h1 class="text-3xl sm:text-5xl font-bold tracking-tight leading-none">MC Report</h1>
      <p class="text-sm text-slate-500 mt-2">
        Field odds, seed odds, and optional season-by-season logs. Outdir: <span class="font-mono text-slate-600">{outdir.as_posix()}</span>
      </p>
    </div>

    <!-- Overview Card -->
    <div class="bg-white shadow-xl shadow-slate-200/50 rounded-2xl overflow-hidden border border-slate-200 mb-8">
      <div class="px-5 py-4 bg-slate-50 border-b border-slate-200 flex flex-col sm:flex-row gap-3 sm:items-center sm:justify-between">
        <div>
          <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Overview</div>
          <div class="text-sm font-semibold text-slate-800">Sortable probabilities + seed summaries</div>
        </div>

        <div class="flex flex-col sm:flex-row gap-2 sm:items-center w-full sm:w-auto">
          <input id="ovSearch"
                 class="w-full sm:w-72 px-3 py-2 rounded-lg border border-slate-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                 placeholder="Search team..." />
        </div>
      </div>

      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse whitespace-nowrap text-sm" id="ovTable">
          <thead class="bg-slate-50 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 font-bold">
            <tr id="ovHeadRow"></tr>
          </thead>
          <tbody class="divide-y divide-slate-100 bg-white" id="ovBody"></tbody>
        </table>
      </div>

      <div class="px-5 py-3 bg-white border-t border-slate-100 text-[11px] text-slate-400">
        Click any header to sort. <span class="font-mono">AtLargeIfNoAQ_Prob</span> = P(at-large | no AQ).
      </div>
    </div>

    <!-- Seed Explorer + Season summaries -->
    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
      <div class="bg-white shadow-xl shadow-slate-200/50 rounded-2xl border border-slate-200 overflow-hidden">
        <div class="px-5 py-4 bg-slate-50 border-b border-slate-200">
          <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Seed Explorer</div>
          <div class="text-sm font-semibold text-slate-800">P(Seed = k)</div>
        </div>

        <div class="p-5">
          <div class="flex flex-col sm:flex-row gap-2 sm:items-center">
            <input id="seedTeam"
                   class="w-full sm:w-80 px-3 py-2 rounded-lg border border-slate-200 bg-white text-sm focus:outline-none focus:ring-2 focus:ring-brand-500"
                   placeholder="Type a team..." />
            <button id="seedPickTop"
                    class="px-3 py-2 rounded-lg border border-slate-200 bg-white text-sm hover:bg-slate-50">
              Pick Top
            </button>
          </div>
          <div class="mt-4">
            <canvas id="chartSeedDist" height="150"></canvas>
          </div>
          <div class="mt-2 text-[11px] text-slate-400">
            Uses <span class="font-mono">mc_seed_odds_long.csv</span> (Team, Seed, Prob).
          </div>
        </div>
      </div>

      <div class="bg-white shadow-xl shadow-slate-200/50 rounded-2xl border border-slate-200 overflow-hidden">
        <div class="px-5 py-4 bg-slate-50 border-b border-slate-200">
          <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Season Logs</div>
          <div class="text-sm font-semibold text-slate-800">Top AQs + At-Large (if JSONL exists)</div>
        </div>

        <div class="p-5">
          <div class="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Top AQs</div>
              <div class="mt-2 overflow-hidden rounded-xl border border-slate-200">
                <div class="overflow-x-auto">
                  <table class="w-full text-left border-collapse whitespace-nowrap text-sm" id="aqTable">
                    <thead class="bg-slate-50 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 font-bold">
                      <tr>
                        <th class="px-4 py-3 cursor-pointer" data-table="aq" data-key="Team">Team</th>
                        <th class="px-4 py-3 text-right cursor-pointer" data-table="aq" data-key="Prob">Prob</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 bg-white" id="aqBody"></tbody>
                  </table>
                </div>
              </div>
            </div>

            <div>
              <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Top At-Large</div>
              <div class="mt-2 overflow-hidden rounded-xl border border-slate-200">
                <div class="overflow-x-auto">
                  <table class="w-full text-left border-collapse whitespace-nowrap text-sm" id="alTable">
                    <thead class="bg-slate-50 border-b border-slate-200 text-xs uppercase tracking-wider text-slate-500 font-bold">
                      <tr>
                        <th class="px-4 py-3 cursor-pointer" data-table="al" data-key="Team">Team</th>
                        <th class="px-4 py-3 text-right cursor-pointer" data-table="al" data-key="Prob">Prob</th>
                      </tr>
                    </thead>
                    <tbody class="divide-y divide-slate-100 bg-white" id="alBody"></tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>

          <div class="mt-5">
            <div class="text-xs font-bold text-slate-400 uppercase tracking-widest">Season samples</div>
            <div class="mt-2 max-h-64 overflow-auto border border-slate-200 rounded-xl bg-slate-50">
              <pre id="seasonSamples" class="p-4 text-[11px] text-slate-700 font-mono whitespace-pre-wrap"></pre>
            </div>
            <div class="mt-2 text-[11px] text-slate-400">
              Up to 25 JSONL entries across detected log files.
            </div>
          </div>
        </div>
      </div>
    </div>

  </main>

  <footer class="mt-auto py-8 text-center border-t border-slate-200 bg-white">
    <p class="text-slate-400 text-xs font-mono">CHan Analytics • MC report generator</p>
  </footer>

  <script>
    // ----------------------------
    // Embedded data
    // ----------------------------
    const FIELD_ODDS = {json.dumps(js_field)};
    const OVERVIEW   = {json.dumps(js_overview)};
    const SEED_LONG  = {json.dumps(js_seed_long)};
    const SEASON_SAMPLES = {json.dumps(season_samples)};
    const AQ_SUM = {json.dumps(js_aq)};
    const AL_SUM = {json.dumps(js_al)};

    // ----------------------------
    // Navbar helpers
    // ----------------------------
    function toggleMenu() {{
      const menu = document.getElementById('mobile-menu');
      if (!menu) return;
      menu.classList.toggle('open');
    }}
    document.addEventListener("click", (e) => {{
      const menu = document.getElementById("mobile-menu");
      if (!menu || !menu.classList.contains("open")) return;
      const a = e.target.closest("#mobile-menu a");
      if (a) menu.classList.remove("open");
    }});
    window.addEventListener("resize", () => {{
      const menu = document.getElementById("mobile-menu");
      if (!menu) return;
      if (window.innerWidth >= 768) menu.classList.remove("open");
    }});
    function scrollToTop(e) {{
      if (e) e.preventDefault();
      const anchor = document.getElementById("topAnchor");
      if (!anchor) return;
      const nav = document.getElementById("siteNav");
      const navH = nav ? nav.getBoundingClientRect().height : 0;
      const pad = 12;
      const y = anchor.getBoundingClientRect().top + window.pageYOffset - navH - pad;
      window.scrollTo({{ top: Math.max(0, y), behavior: "smooth" }});
      history.replaceState(null, "", window.location.pathname);
    }}

    // ----------------------------
    // Sorting utilities (robust)
    // ----------------------------
    function isNum(v) {{
      const x = Number(v);
      return Number.isFinite(x);
    }}
    function cmp(a, b) {{
      // numeric first if both numeric, else string
      const an = Number(a), bn = Number(b);
      if (Number.isFinite(an) && Number.isFinite(bn)) {{
        return an < bn ? -1 : (an > bn ? 1 : 0);
      }}
      const as = String(a ?? "");
      const bs = String(b ?? "");
      return as.localeCompare(bs);
    }}
    function fmtProb(x) {{
      const v = Number(x);
      if (!Number.isFinite(v)) return "—";
      return (100*v).toFixed(1) + "%";
    }}
    function esc(s) {{
      return String(s ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
    }}

    // ----------------------------
    // Overview table (dynamic columns, clickable headers)
    // ----------------------------
    const DEFAULT_OV_COLS = [
      "Team",
      "MakeField_Prob",
      "AtLarge_Prob",
      "AutoBid_Prob",
      "AtLargeIfNoAQ_Prob",
      "BestSeed",
      "BestSeed_Prob",
      "ModeSeed",
      "ModeSeed_Prob"
    ];

    let ovState = {{
      q: "",
      key: "MakeField_Prob",
      dir: "desc"
    }};

    function getOverviewRows() {{
      const base = (OVERVIEW && OVERVIEW.length) ? OVERVIEW.slice() : (FIELD_ODDS || []).slice();
      const q = (ovState.q || "").trim().toLowerCase();
      let rows = base;

      if (q) rows = rows.filter(r => String(r.Team || "").toLowerCase().includes(q));

      const key = ovState.key;
      const dir = ovState.dir;

      rows.sort((ra, rb) => {{
        const c = cmp(ra[key], rb[key]);
        return dir === "asc" ? c : -c;
      }});

      return rows;
    }}

    function resolveOverviewColumns() {{
      // show only columns that exist in at least one row
      const base = (OVERVIEW && OVERVIEW.length) ? OVERVIEW : FIELD_ODDS;
      if (!base || base.length === 0) return ["Team"];
      const present = new Set();
      for (const r of base) {{
        for (const k of Object.keys(r)) present.add(k);
      }}
      const cols = [];
      for (const k of DEFAULT_OV_COLS) {{
        if (present.has(k)) cols.push(k);
      }}
      // ensure Team first
      if (!cols.includes("Team") && present.has("Team")) cols.unshift("Team");
      return cols.length ? cols : ["Team"];
    }}

    function headerLabel(k) {{
      const map = {{
        "MakeField_Prob": "Make",
        "AtLarge_Prob": "At-Large",
        "AutoBid_Prob": "Auto",
        "AtLargeIfNoAQ_Prob": "AtLg | NoAQ",
        "NoAQ_Prob": "NoAQ",
        "BestSeed": "Best Seed",
        "BestSeed_Prob": "Best Seed P",
        "ModeSeed": "Mode Seed",
        "ModeSeed_Prob": "Mode Seed P",
        "Team": "Team"
      }};
      return map[k] || k;
    }}

    function renderOverview() {{
      const head = document.getElementById("ovHeadRow");
      const body = document.getElementById("ovBody");
      if (!head || !body) return;

      const cols = resolveOverviewColumns();

      head.innerHTML = cols.map(k => {{
        const isActive = (ovState.key === k);
        const arrow = isActive ? (ovState.dir === "asc" ? " ↑" : " ↓") : "";
        const align = (k === "Team") ? "" : " text-right";
        const pad = (k === "Team") ? "px-5 py-4" : "px-5 py-4";
        return `<th class="${{pad}}${{align}} cursor-pointer select-none"
                    onclick="setOvSort('${{k}}')">${{headerLabel(k)}}${{arrow}}</th>`;
      }}).join("");

      const rows = getOverviewRows();

      body.innerHTML = rows.map(r => {{
        const cells = cols.map(k => {{
          let v = r[k];
          let cls = "px-5 py-4";
          let content = "";

          if (k === "Team") {{
            content = `<span class="font-semibold text-slate-900">${{esc(v)}}</span>`;
          }} else if (String(k).endsWith("_Prob")) {{
            content = `<span class="font-mono font-bold text-slate-900">${{fmtProb(v)}}</span>`;
            cls += " text-right";
          }} else if (String(k).includes("Prob")) {{
            // for odd prob columns not ending _Prob
            content = `<span class="font-mono text-slate-700">${{fmtProb(v)}}</span>`;
            cls += " text-right";
          }} else if (k === "BestSeed" || k === "ModeSeed") {{
            content = `<span class="font-mono text-slate-700">${{(v===null||v===undefined)?'—':esc(v)}}</span>`;
            cls += " text-right";
          }} else {{
            // fallback
            content = `<span class="font-mono text-slate-700">${{(v===null||v===undefined)?'—':esc(v)}}</span>`;
            cls += " text-right";
          }}

          return `<td class="${{cls}}">${{content}}</td>`;
        }}).join("");

        return `<tr class="hover:bg-slate-50 transition-colors">${{cells}}</tr>`;
      }}).join("") || `<tr><td colspan="${{cols.length}}" class="px-5 py-12 text-center text-slate-400">No results.</td></tr>`;
    }}

    function setOvSort(k) {{
      if (ovState.key === k) {{
        ovState.dir = (ovState.dir === "desc") ? "asc" : "desc";
      }} else {{
        ovState.key = k;
        // default directions: Team asc, numeric desc
        ovState.dir = (k === "Team") ? "asc" : "desc";
      }}
      renderOverview();
    }}

    function initOverviewControls() {{
      const inp = document.getElementById("ovSearch");
      if (inp) {{
        inp.addEventListener("input", () => {{
          ovState.q = inp.value || "";
          renderOverview();
        }});
      }}
    }}

    // ----------------------------
    // AQ / At-large tables (sortable by clicking headers)
    // ----------------------------
    let smallState = {{
      aq: {{ key: "Prob", dir: "desc" }},
      al: {{ key: "Prob", dir: "desc" }}
    }};

    function sortRows(rows, key, dir) {{
      const out = (rows || []).slice();
      out.sort((a,b) => {{
        const c = cmp(a[key], b[key]);
        return dir === "asc" ? c : -c;
      }});
      return out;
    }}

    function renderSmallTable(kind, rows) {{
      const bodyId = (kind === "aq") ? "aqBody" : "alBody";
      const body = document.getElementById(bodyId);
      if (!body) return;

      const st = smallState[kind];
      const sorted = sortRows(rows || [], st.key, st.dir).slice(0, 25);

      body.innerHTML = sorted.map(r => {{
        return `
          <tr class="hover:bg-slate-50 transition-colors">
            <td class="px-4 py-3 font-semibold text-slate-900">${{esc(r.Team)}}</td>
            <td class="px-4 py-3 text-right font-mono text-slate-700 font-bold">${{fmtProb(r.Prob)}}</td>
          </tr>
        `;
      }}).join("") || `<tr><td colspan="2" class="px-4 py-10 text-center text-slate-400">No data.</td></tr>`;
    }}

    function onSmallHeaderClick(kind, key) {{
      const st = smallState[kind];
      if (st.key === key) st.dir = (st.dir === "desc") ? "asc" : "desc";
      else {{
        st.key = key;
        st.dir = (key === "Team") ? "asc" : "desc";
      }}
      renderSmallTable(kind, kind === "aq" ? AQ_SUM : AL_SUM);
      // update header arrows
      updateSmallHeaderArrows();
    }}

    function updateSmallHeaderArrows() {{
      for (const el of document.querySelectorAll("[data-table]")) {{
        const kind = el.getAttribute("data-table");
        const key = el.getAttribute("data-key");
        const st = smallState[kind];
        const base = key;
        const label = (key === "Prob") ? "Prob" : "Team";
        const arrow = (st.key === key) ? (st.dir === "asc" ? " ↑" : " ↓") : "";
        el.textContent = label + arrow;
      }}
    }}

    function initSmallTables() {{
      // delegate clicks
      document.querySelectorAll("[data-table]").forEach(el => {{
        el.addEventListener("click", () => {{
          const kind = el.getAttribute("data-table");
          const key = el.getAttribute("data-key");
          onSmallHeaderClick(kind, key);
        }});
      }});
      updateSmallHeaderArrows();
      renderSmallTable("aq", AQ_SUM);
      renderSmallTable("al", AL_SUM);
    }}

    // ----------------------------
    // Season samples
    // ----------------------------
    function renderSeasonSamples() {{
      const el = document.getElementById("seasonSamples");
      if (!el) return;
      if (!SEASON_SAMPLES || SEASON_SAMPLES.length === 0) {{
        el.textContent = "No JSONL season logs detected in outdir.";
        return;
      }}
      const pretty = SEASON_SAMPLES.map(s => {{
        const hdr = `# source=${{s.source}}  idx=${{s.i}}`;
        let body = "";
        try {{ body = JSON.stringify(s.json, null, 2); }}
        catch(e) {{ body = String(s.json); }}
        return hdr + "\\n" + body;
      }}).join("\\n\\n");
      el.textContent = pretty;
    }}

    // ----------------------------
    // Seed chart
    // ----------------------------
    let chartSeed = null;

    function seedDistForTeam(teamName) {{
      const t = (teamName || "").trim().toLowerCase();
      if (!t) return null;

      const allTeams = Array.from(new Set((SEED_LONG || []).map(r => String(r.Team || "")))).filter(Boolean);
      if (!allTeams.length) return null;

      let pick = allTeams.find(x => x.toLowerCase() === t);
      if (!pick) pick = allTeams.find(x => x.toLowerCase().includes(t));
      if (!pick) return null;

      const probs = new Array(16).fill(0);
      for (const r of SEED_LONG) {{
        if (String(r.Team || "") !== pick) continue;
        const seed = Number(r.Seed);
        const p = Number(r.Prob);
        if (Number.isFinite(seed) && seed >= 1 && seed <= 16 && Number.isFinite(p)) {{
          probs[seed-1] += p;
        }}
      }}

      return {{ team: pick, probs }};
    }}

    function renderSeedChart(teamName) {{
      const canvas = document.getElementById("chartSeedDist");
      if (!canvas) return;

      const dist = seedDistForTeam(teamName);

      const labels = Array.from({{length: 16}}, (_,i)=>String(i+1));
      const data = dist ? dist.probs.map(p => 100*p) : new Array(16).fill(0);
      const title = dist ? `Seed distribution: ${{dist.team}}` : "Seed distribution (no match)";

      if (chartSeed) chartSeed.destroy();
      chartSeed = new Chart(canvas, {{
        type: "bar",
        data: {{
          labels,
          datasets: [{{ label: "P(Seed)", data }}]
        }},
        options: {{
          responsive: true,
          plugins: {{
            legend: {{ display: false }},
            title: {{ display: true, text: title }}
          }},
          scales: {{
            y: {{
              beginAtZero: true,
              ticks: {{ callback: (v) => v + "%" }}
            }}
          }}
        }}
      }});
    }}

    function initSeedControls() {{
      const inp = document.getElementById("seedTeam");
      const btn = document.getElementById("seedPickTop");

      if (inp) inp.addEventListener("input", () => renderSeedChart(inp.value || ""));

      if (btn) btn.addEventListener("click", () => {{
        const rows = (FIELD_ODDS || []).slice();
        rows.sort((a,b) => Number(b.MakeField_Prob||0) - Number(a.MakeField_Prob||0));
        const top = rows[0];
        if (top && inp) {{
          inp.value = top.Team || "";
          renderSeedChart(inp.value);
        }}
      }});

      // initial team = top MakeField team
      const rows = (FIELD_ODDS || []).slice();
      rows.sort((a,b) => Number(b.MakeField_Prob||0) - Number(a.MakeField_Prob||0));
      const top = rows[0];
      if (top && inp) inp.value = top.Team || "";
      renderSeedChart(inp ? inp.value : "");
    }}

    // ----------------------------
    // Init
    // ----------------------------
    initOverviewControls();
    renderOverview();
    initSmallTables();
    renderSeasonSamples();
    initSeedControls();
  </script>
</body>
</html>
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="mc_out", help="Directory containing MC outputs.")
    ap.add_argument("--html", default="mc_report.html", help="Output HTML file path.")
    ap.add_argument("--max_seasons", type=int, default=2000, help="Max JSONL seasons to load per file.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    if not outdir.exists():
        raise FileNotFoundError(f"outdir not found: {outdir}")

    df_field = read_csv_if_exists(outdir / "mc_field_odds.csv")
    df_seed_wide = read_csv_if_exists(outdir / "mc_seed_odds_wide.csv")
    df_seed_long = read_csv_if_exists(outdir / "mc_seed_odds_long.csv")

    season_logs: List[Tuple[str, List[Dict[str, Any]]]] = []
    jsonl_paths = find_jsonl_logs(outdir)
    for p in jsonl_paths:
        seasons = load_jsonl(p, max_lines=args.max_seasons)
        if seasons:
            season_logs.append((p.name, seasons))

    all_seasons: List[Dict[str, Any]] = []
    for _name, seasons in season_logs:
        all_seasons.extend(seasons)
    season_summary = summarize_seasons(all_seasons)

    html = build_html(
        outdir=outdir,
        df_field=df_field,
        df_seed_wide=df_seed_wide,
        df_seed_long=df_seed_long,
        season_logs=season_logs,
        season_summary=season_summary,
        title="MC Report | CHan Analytics",
    )

    out_path = Path(args.html)
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML report: {out_path.resolve()}")


if __name__ == "__main__":
    main()
