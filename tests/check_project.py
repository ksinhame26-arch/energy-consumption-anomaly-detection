"""Project health check. Run from the project folder AFTER the three scripts:
    python src/run_pipeline.py
    python src/run_weather_experiment.py
    python src/build_dashboard.py
    python tests/check_project.py

Checks data integrity, internal consistency of every output, that the headline numbers in
README.md match the result files, and that all figures and the dashboard exist.
Prints PASS/FAIL per check and exits with code 1 if anything fails.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))

results = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


# ------------------------------------------------------------------ data
raw = pd.read_csv(ROOT / "data/raw/PJME_hourly.csv")
check("raw load file has 145,366 rows", len(raw) == 145366, f"{len(raw):,}")
clean = pd.read_csv(ROOT / "data/processed/pjme_clean.csv", index_col=0, parse_dates=True)
check("clean series has 145,392 hourly rows", len(clean) == 145392, f"{len(clean):,}")
check("clean index has no duplicates", clean.index.is_unique)
check("clean index is a regular hourly grid",
      (clean.index.to_series().diff().dropna() == pd.Timedelta(hours=1)).all())
check("clean load has no missing values", clean["load_mw"].notna().all())
check("exactly 30 hours were imputed", int(clean["imputed"].sum()) == 30, str(int(clean["imputed"].sum())))
check("load values are physically plausible (10-70 GW)",
      clean["load_mw"].between(10000, 70000).all(),
      f"min {clean['load_mw'].min():.0f}, max {clean['load_mw'].max():.0f}")

from weather import load_temperature  # noqa: E402
temp = load_temperature()
check("temperature series is hourly, Oct 2012 - Nov 2017",
      str(temp.index.min())[:7] == "2012-10" and str(temp.index.max())[:7] == "2017-11")
check("temperature values plausible for Philadelphia (-25..42 C)",
      temp.dropna().between(-25, 42).all(), f"{temp.min():.1f}..{temp.max():.1f}")
summer = temp[temp.index.month.isin([6, 7, 8])]
hourly = summer.groupby(summer.index.hour).mean()
check("timezone conversion correct (summer coolest ~5-7h, warmest ~13-17h local)",
      hourly.idxmin() in (4, 5, 6, 7) and hourly.idxmax() in (13, 14, 15, 16, 17),
      f"coolest {hourly.idxmin()}h, warmest {hourly.idxmax()}h")

# ------------------------------------------------------------------ main pipeline output
res = pd.read_csv(OUT / "anomalies_hourly.csv", index_col=0, parse_dates=True)
check("hourly results cover every hour", len(res) == len(clean) and (res.index == clean.index).all())
flag_cols = [c for c in res.columns if c.endswith("_flag")]
check("no missing values in any flag column", res[flag_cols].notna().all().all())
check("votes = number of the 4 detectors that fired",
      (res["votes"] == res[["stl_flag", "fc_flag", "if_flag", "ae_flag"]].sum(axis=1)).all())
check("ensemble flag = votes >= 2", (res["ens_flag"] == (res["votes"] >= 2)).all())
check("final flag = ensemble OR rules", (res["final_flag"] == (res["ens_flag"] | res["dq_flag"])).all())
check("forecast model produced a prediction for every hour", res["fc_expected"].notna().all())
check("Isolation Forest flags ~1% of hours (its contamination setting)",
      abs(res["if_flag"].mean() - 0.01) < 0.001, f"{res['if_flag'].mean():.4f}")
n_final = int(res["final_flag"].sum())
summary = json.loads((OUT / "run_summary.json").read_text())
check("run_summary.json agrees with hourly results", summary["flagged_hours"]["final"] == n_final)
sandy = res.loc["2012-10-29":"2012-10-30"]
check("Hurricane Sandy is flagged by all 4 detectors",
      all(sandy[f"{m}_flag"].any() for m in ["stl", "fc", "if", "ae"]))
check("Hurricane Sandy (29-30 Oct 2012) has at least 10 anomalous hours",
      int(sandy["final_flag"].sum()) >= 10, f"{int(sandy['final_flag'].sum())} hours")

# ------------------------------------------------------------------ evaluation
ev = pd.read_csv(OUT / "injected_events.csv", parse_dates=["start", "end"])
check("50 injected events, 10 per type", len(ev) == 50 and (ev["type"].value_counts() == 10).all())
check("injected events lie inside the 2017 test window",
      (ev["start"] >= "2017-01-01").all() and (ev["end"] <= "2017-12-31 23:00").all())
gaps = (ev["start"].iloc[1:].values - ev["end"].iloc[:-1].values) / np.timedelta64(1, "h")
check("injected events do not overlap (>= 72h apart)", (gaps >= 72).all(), f"min gap {gaps.min():.0f}h")
check("event lengths match their type",
      ((ev["end"] - ev["start"]) / pd.Timedelta(hours=1) + 1 == ev["length_h"]).all())


def f1_consistent(df):
    p, r, f = df["point_precision"], df["point_recall"], df["point_f1"]
    calc = np.where(p + r > 0, 2 * p * r / (p + r).replace(0, 1), 0)
    return np.allclose(calc, f, atol=0.002)


em = pd.read_csv(OUT / "evaluation_metrics.csv")
check("evaluation: F1 = 2PR/(P+R) for every method", f1_consistent(em))
check("evaluation: all metrics within [0, 1]",
      em[["point_precision", "point_recall", "point_f1", "event_recall"]].stack().between(0, 1).all())
bt = pd.read_csv(OUT / "evaluation_by_type.csv")
mean_by_type = bt.groupby("method")["event_recall"].mean()
em_short = em.set_index("method")["event_recall"]
check("per-type recall averages to overall event recall (equal-size types)",
      np.isclose(mean_by_type["final"], em_short["Final (ensemble OR rules)"], atol=0.001))

wi = pd.read_csv(OUT / "weather_experiment/injection_metrics_all_methods.csv")
check("weather experiment: F1 consistent", f1_consistent(wi))
wr = pd.read_csv(OUT / "weather_experiment/real_data_comparison.csv", index_col=0)
check("weather model reduces forecast error",
      wr.loc["Calendar + temperature", "MAPE_%"] < wr.loc["Calendar only", "MAPE_%"])
check("weather model still catches Hurricane Sandy",
      wr.loc["Calendar + temperature", "flags_Hurricane Sandy"] > 0)

# ------------------------------------------------------------------ README numbers
readme = (ROOT / "README.md").read_text(encoding="utf-8")
fin = em.set_index("method").loc["Final (ensemble OR rules)"]
wf = wi[wi["method"] == "final"].set_index("variant")
cal, wx = wr.loc["Calendar only"], wr.loc["Calendar + temperature"]
claims = {  # exact table rows / sentences from README.md
    "main results row": f"| **Final (ensemble OR rules)** | **{fin['point_precision']:.2f}** | "
                        f"**{fin['point_recall']:.2f}** | **{fin['point_f1']:.2f}** | **{fin['event_recall']:.2f}** |",
    "flagged hours": f"- {n_final} anomalous hours flagged ({100 * n_final / len(res):.2f}% of all hours)",
    "MAPE row": f"| Forecast error (MAPE) | {cal['MAPE_%']:.1f}% | **{wx['MAPE_%']:.1f}%** |",
    "RMSE row": f"| Forecast error (RMSE) | {cal['RMSE_MW']:,.0f} MW | **{wx['RMSE_MW']:,.0f} MW** |",
    "flags row": f"| Hours flagged on real data (final detector) | {int(cal['final_flags'])} | **{int(wx['final_flags'])}** |",
    "extreme-day row": f"| {int(cal['final_flags_on_extreme_temp_days'])} | **{int(wx['final_flags_on_extreme_temp_days'])}** |",
    "precision row": f"| Precision | {wf.loc['calendar', 'point_precision']:.2f} | **{wf.loc['weather', 'point_precision']:.2f}** |",
    "recall row": f"| Recall | {wf.loc['calendar', 'point_recall']:.2f} | **{wf.loc['weather', 'point_recall']:.2f}** |",
    "F1 row": f"| F1 | {wf.loc['calendar', 'point_f1']:.2f} | **{wf.loc['weather', 'point_f1']:.2f}** |",
    "event recall row": f"| Event recall | {wf.loc['calendar', 'event_recall']:.2f} | **{wf.loc['weather', 'event_recall']:.2f}** |",
}
for name, val in claims.items():
    check(f"README {name} matches results", val in readme, val if val not in readme else "")

# ------------------------------------------------------------------ figures & dashboard
figs = sorted((OUT / "figures").glob("*.png")) + sorted((OUT / "weather_experiment/figures").glob("*.png"))
check("all 15 figures exist and are non-trivial (>10 KB)",
      len(figs) == 15 and all(f.stat().st_size > 10_000 for f in figs), f"{len(figs)} found")
dash = OUT / "dashboard.html"
check("dashboard.html exists", dash.exists())
if dash.exists():
    html = dash.read_text(encoding="utf-8")
    m = re.search(r"const D = (\{.*?\});\n", html, re.S)
    check("dashboard data block present (template placeholder replaced)",
          m is not None and "__DATA__" not in html)
    if m:
        d = json.loads(m.group(1))
        check("dashboard: hour count matches results", d["n"] == len(res) == len(d["load"]))
        check("dashboard: anomaly count matches results",
              d["kpi"]["final_flags"] == n_final and sum(1 for f in d["fl"] if f & 32) == n_final)
        check("dashboard: weather block aligned with weather period",
              len(d["wx"]["exp"]) == len(d["wx"]["fl"]) == len(d["wx"]["temp"]))
        check("dashboard: no external scripts or fonts (works offline)",
              "<script src" not in html and "http" not in html.split("<script>")[1][:200])

failed = [r for r in results if not r[1]]
print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
sys.exit(1 if failed else 0)
