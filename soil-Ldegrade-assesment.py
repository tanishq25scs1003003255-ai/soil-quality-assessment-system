"""
soil_degradation_assessment.py

Usage:
- Place a CSV with columns: site, pH, EC_dS_m, OM_pct, N_mgkg, P_mgkg, K_mgkg, BD_gcc, erosion_score, veg_cover_pct
- If no CSV provided or file not found, synthetic sample data will be used.

Outputs:
- 'soil_quality_results.csv' with normalized indicators, SQI, DI, class, recommendations, projected_SQI
- Plots saved as PNG files in current directory
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Utility: generate sample data if no CSV ----------
def generate_sample_data(n_sites=12, seed=42):
    np.random.seed(seed)
    sites = [f"Site_{i+1}" for i in range(n_sites)]
    # realistic ranges for indicators (approximate)
    pH = np.random.normal(6.2, 0.9, n_sites)          # 4.5 - 8.5
    EC = np.abs(np.random.normal(0.6, 0.7, n_sites))  # dS/m (salinity)
    OM = np.clip(np.random.normal(2.5, 1.5, n_sites), 0.1, 10.0)  # %
    N = np.clip(np.random.normal(2500, 900, n_sites), 50, 8000)  # mg/kg (or kg/ha equiv)
    P = np.clip(np.random.normal(12, 8, n_sites), 1, 200)        # mg/kg
    K = np.clip(np.random.normal(120, 80, n_sites), 10, 800)     # mg/kg
    BD = np.clip(np.random.normal(1.45, 0.18, n_sites), 0.9, 2.0) # g/cm3
    erosion = np.clip(np.round(np.random.uniform(0,5,n_sites)), 0,5) # 0 (none) to 5 (severe)
    veg = np.clip(np.random.normal(45, 25, n_sites), 0, 100)    # % vegetation cover

    df = pd.DataFrame({
        "site": sites,
        "pH": pH,
        "EC_dS_m": EC,
        "OM_pct": OM,
        "N_mgkg": N,
        "P_mgkg": P,
        "K_mgkg": K,
        "BD_gcc": BD,
        "erosion_score": erosion,
        "veg_cover_pct": veg
    })
    return df

# ---------- Normalization/scoring for each indicator ----------
# Each function returns score between 0 and 1 where 1 is ideal.

def score_pH(pH):
    # optimum around 6.5-7.5 for many crops; penalize too acidic or too alkaline
    # we map: 6.5-7.5 -> 1.0 ; linear drop to 0 at 4.5 and 9.0
    if np.isnan(pH): return 0.5
    if 6.5 <= pH <= 7.5:
        return 1.0
    elif pH < 6.5:
        return max(0.0, (pH - 4.5) / (6.5 - 4.5))
    else:
        return max(0.0, (9.0 - pH) / (9.0 - 7.5))

def score_EC(ec):
    # lower EC better (non-saline). EC (dS/m) < 1 good; >4 high salinity.
    if np.isnan(ec): return 0.5
    if ec <= 1.0:
        return 1.0
    elif ec >= 4.0:
        return 0.0
    else:
        return max(0.0, (4.0 - ec) / (4.0 - 1.0))

def score_OM(om):
    # organic matter: >4% excellent, 2-4 moderate, <1 poor
    if np.isnan(om): return 0.2
    if om >= 4.0:
        return 1.0
    elif om <= 0.5:
        return 0.0
    else:
        return (om - 0.5) / (4.0 - 0.5)

def score_N(n):
    # available N (mg/kg) - rough scaling: >3000 good, <500 poor
    if np.isnan(n): return 0.2
    n = float(n)
    if n >= 3000:
        return 1.0
    elif n <= 500:
        return 0.0
    else:
        return (n - 500) / (3000 - 500)

def score_P(p):
    # Olsen P mg/kg: >15 good, <5 poor
    if np.isnan(p): return 0.2
    p = float(p)
    if p >= 20:
        return 1.0
    elif p <= 3:
        return 0.0
    else:
        return (p - 3) / (20 - 3)

def score_K(k):
    # K mg/kg: >150 good, <50 poor
    if np.isnan(k): return 0.2
    k = float(k)
    if k >= 150:
        return 1.0
    elif k <= 30:
        return 0.0
    else:
        return (k - 30) / (150 - 30)

def score_BD(bd):
    # bulk density: lower is better up to a point. For many soils: <1.3 good, >1.6 poor for topsoil
    if np.isnan(bd): return 0.5
    if bd <= 1.3:
        return 1.0
    elif bd >= 1.7:
        return 0.0
    else:
        return max(0.0, (1.7 - bd) / (1.7 - 1.3))

def score_erosion(e):
    # erosion: 0 best, 5 worst.
    if np.isnan(e): return 0.5
    e = float(e)
    if e <= 0:
        return 1.0
    elif e >= 5:
        return 0.0
    else:
        return max(0.0, (5.0 - e) / 5.0)

def score_veg(veg):
    # vegetation cover %: higher is better
    if np.isnan(veg): return 0.2
    veg = float(veg)
    if veg >= 80:
        return 1.0
    elif veg <= 5:
        return 0.0
    else:
        return veg / 80.0

# ---------- Weights for indicators (must sum to 1) ----------
WEIGHTS = {
    "OM_pct": 0.20,
    "N_mgkg": 0.12,
    "P_mgkg": 0.10,
    "K_mgkg": 0.10,
    "pH": 0.08,
    "EC_dS_m": 0.08,
    "BD_gcc": 0.10,
    "erosion_score": 0.12,
    "veg_cover_pct": 0.10
}

# ---------- Compute scores, SQI, DI, class ----------
def compute_scores(df):
    # apply scoring functions
    df = df.copy()
    df["score_pH"] = df["pH"].apply(score_pH)
    df["score_EC"] = df["EC_dS_m"].apply(score_EC)
    df["score_OM"] = df["OM_pct"].apply(score_OM)
    df["score_N"] = df["N_mgkg"].apply(score_N)
    df["score_P"] = df["P_mgkg"].apply(score_P)
    df["score_K"] = df["K_mgkg"].apply(score_K)
    df["score_BD"] = df["BD_gcc"].apply(score_BD)
    df["score_erosion"] = df["erosion_score"].apply(score_erosion)
    df["score_veg"] = df["veg_cover_pct"].apply(score_veg)

    # compute SQI weighted
    score_cols = {
        "OM_pct": "score_OM",
        "N_mgkg": "score_N",
        "P_mgkg": "score_P",
        "K_mgkg": "score_K",
        "pH": "score_pH",
        "EC_dS_m": "score_EC",
        "BD_gcc": "score_BD",
        "erosion_score": "score_erosion",
        "veg_cover_pct": "score_veg"
    }

    df["SQI"] = 0.0
    for key, w in WEIGHTS.items():
        df["SQI"] += df[score_cols[key]] * w

    df["DI"] = 1.0 - df["SQI"]  # degradation index
    df["SQI"] = df["SQI"].round(4)
    df["DI"] = df["DI"].round(4)

    # classification
    def classify(sqi):
        if sqi >= 0.80: return "Healthy"
        elif sqi >= 0.60: return "Moderately healthy"
        elif sqi >= 0.40: return "Moderately degraded"
        elif sqi >= 0.20: return "Severely degraded"
        else: return "Extremely degraded"

    df["class"] = df["SQI"].apply(classify)
    return df

# ---------- Diagnostic flags and recommendations ----------
RECOMMENDATION_LIBRARY = {
    "low_OM": "Apply compost/green manures, encourage cover crops, reduce burning. Start residue retention and/or add biochar.",
    "low_N": "Introduce legume rotations, apply organic N sources or targeted mineral N with split application.",
    "low_P": "Apply rock phosphate or triple superphosphate in banded placement; consider mycorrhizal inoculation.",
    "low_K": "Apply potash or organic K sources (compost, ashes) and include K-friendly crops.",
    "acidic": "Apply lime to raise pH (rate based on buffering capacity) + add organic matter.",
    "alkaline": "Use gypsum if sodic; add organic matter and sulfur where appropriate.",
    "high_EC": "Improve drainage/leaching where possible, use salt-tolerant crops, avoid saline irrigation.",
    "high_BD": "Deep ripping where feasible, increase organic matter, reduce compaction by machinery.",
    "low_veg": "Establish perennial vegetative barriers, mulching, and cover crops to protect soil.",
}

def diagnose_and_recommend(row):
    recs = []
    # thresholds in raw or normalized form; we use normalized scores for consistency
    if row["score_OM"] < 0.4:
        recs.append(("low_OM", RECOMMENDATION_LIBRARY["low_OM"]))
    if row["score_N"] < 0.35:
        recs.append(("low_N", RECOMMENDATION_LIBRARY["low_N"]))
    if row["score_P"] < 0.35:
        recs.append(("low_P", RECOMMENDATION_LIBRARY["low_P"]))
    if row["score_K"] < 0.35:
        recs.append(("low_K", RECOMMENDATION_LIBRARY["low_K"]))
    # pH rules
    if row["pH"] < 6.0:
        recs.append(("acidic", RECOMMENDATION_LIBRARY["acidic"]))
    elif row["pH"] > 8.0:
        recs.append(("alkaline", RECOMMENDATION_LIBRARY["alkaline"]))
    # salinity
    if row["score_EC"] < 0.5:
        recs.append(("high_EC", RECOMMENDATION_LIBRARY["high_EC"]))
    if row["score_BD"] < 0.5:
        recs.append(("high_BD", RECOMMENDATION_LIBRARY["high_BD"]))
    if row["score_veg"] < 0.4 or row["erosion_score"] >= 3:
        recs.append(("low_veg", RECOMMENDATION_LIBRARY["low_veg"]))

    # if no specific recs: suggest general soil health program
    if len(recs) == 0:
        recs.append(("general", "Maintain crop rotation, residue retention, and integrated nutrient management."))

    # rank recs: by severity (lower score indicates more severe)
    # compute severity as (1 - normalized_score) for flagged indicators where available
    ranked = sorted(recs, key=lambda x: {
        "low_OM": 1 - row["score_OM"],
        "low_N": 1 - row["score_N"],
        "low_P": 1 - row["score_P"],
        "low_K": 1 - row["score_K"],
        "acidic": max(0.0, (6.0 - row["pH"])) / 4.0,
        "alkaline": max(0.0, (row["pH"] - 8.0)) / 4.0,
        "high_EC": 1 - row["score_EC"],
        "high_BD": 1 - row["score_BD"],
        "low_veg": 1 - row["score_veg"],
        "general": 0.1
    }[x[0]], reverse=True)

    return [r[1] for r in ranked[:5]]  # top 5 suggestions

# ---------- Projected improvement simulation ----------
def simulate_projected_sqi(row, recommendations):
    # crude projection: each recommendation raises SQI by small fraction depending on type
    delta = 0.0
    for rec in recommendations:
        # map keywords to expected delta
        if "compost" in rec or "green manure" in rec or "cover" in rec:
            delta += 0.12  # organic matter build-up over 1-2 years
        if "legume" in rec or "N sources" in rec:
            delta += 0.08
        if "rock phosphate" in rec or "phosphate" in rec:
            delta += 0.06
        if "potash" in rec or "K " in rec or "organic K" in rec:
            delta += 0.05
        if "lime" in rec or "gypsum" in rec:
            delta += 0.06
        if "leaching" in rec or "salt-tolerant" in rec:
            delta += 0.04
        if "deep ripping" in rec or "compaction" in rec:
            delta += 0.07
        if "vegetative" in rec or "mulching" in rec:
            delta += 0.08
    # cap improvements so SQI<=0.98
    new_sqi = min(0.98, row["SQI"] + delta)
    return round(new_sqi, 4)

# ---------- Plotting helpers ----------
def plot_sqi_bar(df):
    df_sorted = df.sort_values("SQI", ascending=False)
    plt.figure(figsize=(10,5))
    bars = plt.bar(df_sorted["site"], df_sorted["SQI"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Soil Quality Index (SQI)")
    plt.ylim(0,1)
    plt.title("SQI per Site")
    # color by class
    color_map = {
        "Healthy": (0.2,0.7,0.2),
        "Moderately healthy": (0.4,0.8,0.6),
        "Moderately degraded": (0.9,0.8,0.3),
        "Severely degraded": (0.95,0.5,0.2),
        "Extremely degraded": (0.8,0.2,0.2)
    }
    for bar, cls in zip(bars, df_sorted["class"]):
        bar.set_color(color_map.get(cls, (0.6,0.6,0.6)))
    plt.tight_layout()
    plt.savefig("sqi_per_site.png", dpi=200)
    plt.close()

def plot_indicator_radar(df, site, indicators=None):
    # radar plot for a single site comparing normalized indicator scores
    if indicators is None:
        indicators = ["score_OM","score_N","score_P","score_K","score_pH","score_EC","score_BD","score_erosion","score_veg"]
        labels = ["OM","N","P","K","pH","EC","BD","Erosion","Veg"]
    else:
        labels = indicators
    vals = df.loc[df["site"]==site, indicators].values.flatten().tolist()
    # close the loop
    vals += vals[:1]
    angles = np.linspace(0, 2*np.pi, len(labels)+1, endpoint=True)
    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(angles[:-1]*180/np.pi, labels)
    ax.set_ylim(0,1)
    ax.set_title(f"Indicator radar for {site}")
    plt.tight_layout()
    plt.savefig(f"radar_{site}.png", dpi=200)
    plt.close()

def plot_om_vs_sqi(df):
    plt.figure(figsize=(6,5))
    plt.scatter(df["OM_pct"], df["SQI"])
    plt.xlabel("Organic Matter (%)")
    plt.ylabel("SQI")
    plt.title("OM vs SQI")
    plt.tight_layout()
    plt.savefig("om_vs_sqi.png", dpi=200)
    plt.close()

def plot_class_histogram(df):
    plt.figure(figsize=(6,4))
    df["class"].value_counts().reindex(["Healthy","Moderately healthy","Moderately degraded","Severely degraded","Extremely degraded"]).plot(kind='bar')
    plt.ylabel("Number of sites")
    plt.title("Distribution of Degradation Classes")
    plt.tight_layout()
    plt.savefig("degradation_class_hist.png", dpi=200)
    plt.close()

# ---------- Main workflow ----------
def main(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}")
    else:
        print("No CSV provided or file not found — generating synthetic sample data.")
        df = generate_sample_data(n_sites=12)

    # ensure expected columns exist (simple check)
    expected_cols = {"site","pH","EC_dS_m","OM_pct","N_mgkg","P_mgkg","K_mgkg","BD_gcc","erosion_score","veg_cover_pct"}
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Input data missing required columns: {missing}")

    # Step 1: compute scores & SQI
    df_scores = compute_scores(df)

    # Step 2: recommendations
    df_scores["recommendations"] = df_scores.apply(lambda r: diagnose_and_recommend(r), axis=1)
    df_scores["projected_SQI"] = df_scores.apply(lambda r: simulate_projected_sqi(r, r["recommendations"]), axis=1)

    # Step 3: save results
    out_cols = [
        "site","pH","EC_dS_m","OM_pct","N_mgkg","P_mgkg","K_mgkg","BD_gcc","erosion_score","veg_cover_pct",
        "score_pH","score_EC","score_OM","score_N","score_P","score_K","score_BD","score_erosion","score_veg",
        "SQI","DI","class","recommendations","projected_SQI"
    ]
    df_scores[out_cols].to_csv("soil_quality_results.csv", index=False)
    print("Saved 'soil_quality_results.csv'")

    # Step 4: visualizations
    plot_sqi_bar(df_scores)
    print("Saved 'sqi_per_site.png'")
    # radar for top 4 worst sites (lowest SQI)
    worst_sites = df_scores.nsmallest(4, "SQI")["site"].tolist()
    for s in worst_sites:
        plot_indicator_radar(df_scores, s)
        print(f"Saved 'radar_{s}.png'")
    plot_om_vs_sqi(df_scores)
    print("Saved 'om_vs_sqi.png'")
    plot_class_histogram(df_scores)
    print("Saved 'degradation_class_hist.png'")

    # show brief summary
    summary = df_scores[['site','SQI','class','recommendations','projected_SQI']].sort_values('SQI')
    print("\nSummary (lowest SQI first):\n", summary.head(8).to_string(index=False))
    return df_scores

if __name__ == "__main__":
    # if you want to use a CSV, change csv_path to the path of your CSV file
    csv_path = "soil_sample_data.csv"  # e.g., "soil_samples.csv"
    results = main(csv_path)
