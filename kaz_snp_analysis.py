"""
Kazakh SNP Population Analysis — Shin et al. 2021 Style
=========================================================
Input:  only_cardio_freqs3.tsv
        Columns: rsID, chr, location, ref, alt, gene,
                 kaviar_frq, kaviar_number, KAZ, EAS, EUR

Methods (matching Shin et al. BMC Ophthalmology 2021):
  1. Fisher's exact test:
       - Each population vs global frequency  (Shin original approach)
       - KAZ vs EUR  (pairwise)
       - KAZ vs EAS  (pairwise)
  2. Signed log10(p): + = enriched, - = depleted
  3. Bonferroni correction across all SNPs tested
  4. Hierarchical-clustered heatmap (red=enriched, purple=depleted)
  5. Composite Genetic Risk Score (GRS) per population
  6. Correlation of GRS with population-level disease prevalence
"""

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1.  LOAD DATA
# =============================================================================

df = pd.read_csv("only_cardio_freqs3.tsv", sep="\t")

# Standardise column names (strip whitespace)
df.columns = df.columns.str.strip()

# Required columns
REQUIRED = ["rsID", "chr", "location", "ref", "alt", "gene",
            "kaviar_frq", "kaviar_number", "KAZ", "EAS", "EUR"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns: {missing}\nFound: {list(df.columns)}")

# Drop rows where ALL frequency columns are NaN
POPS = ["KAZ", "EAS", "EUR"]
df = df.dropna(subset=["kaviar_frq"] + POPS, how="all").reset_index(drop=True)

# Convert frequency columns to numeric (dots → NaN)
for col in ["kaviar_frq", "kaviar_number"] + POPS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows with any NaN in key freq columns
df = df.dropna(subset=["kaviar_frq", "kaviar_number"] + POPS).reset_index(drop=True)

N_SNPS = len(df)
N_HYPOTHESES = N_SNPS   # Bonferroni over all SNPs tested (like paper uses 138)
ALPHA = 0.05
bonf_thresh = ALPHA / N_HYPOTHESES

print(f"Loaded {N_SNPS} SNPs across populations: {POPS}")
print(f"Bonferroni threshold: p < {bonf_thresh:.2e}  (0.05 / {N_HYPOTHESES})")


# =============================================================================
# 2.  SAMPLE SIZES
#     kaviar_number in the input file is the per-variant allele number (AN).
#     N individuals = AN / 2, computed per-row in the analysis loop.
#     KAZ/EAS/EUR are your cohort sizes.
# =============================================================================

N_SAMPLES = {
    "KAZ": 224,   # Kazakh cohort
    "EAS": 152,   # East Asian reference
    "EUR": 121,   # European reference
}

# =============================================================================
# 3.  FISHER'S EXACT TEST FUNCTIONS
# =============================================================================

def fisher_vs_reference(eaf_pop, n_pop, eaf_ref, n_ref):
    """
    Two-sided Fisher's exact test: population vs reference (global or another pop).
    Returns (p_value, direction): direction +1=enriched, -1=depleted.
    """
    n_pop_alleles = 2 * n_pop
    n_ref_alleles = 2 * n_ref

    k_pop = round(eaf_pop * n_pop_alleles)
    k_ref = round(eaf_ref * n_ref_alleles)

    k_pop = max(0, min(k_pop, n_pop_alleles))
    k_ref = max(0, min(k_ref, n_ref_alleles))

    table = np.array([
        [k_pop, n_pop_alleles - k_pop],
        [k_ref, n_ref_alleles - k_ref],
    ], dtype=float)

    if table.min() < 0 or table.sum() == 0:
        return 1.0, 0

    _, p = fisher_exact(table, alternative="two-sided")
    direction = 1 if eaf_pop >= eaf_ref else -1
    return p, direction


def signed_log10(p, direction):
    p = max(p, 1e-300)
    return -np.log10(p) * direction   # + enriched, - depleted


# =============================================================================
# 4.  RUN TESTS
#
#     For each SNP compute:
#       A) Each population vs GLOBAL  (Shin et al. original approach)
#       B) KAZ vs EUR  (pairwise — your primary question)
#       C) KAZ vs EAS  (pairwise — your primary question)
# =============================================================================

results = []

for _, row in df.iterrows():
    # per-variant Kaviar N: kaviar_number is allele number (AN), so divide by 2
    kaviar_n = int(row["kaviar_number"]) // 2

    snp_result = {
        "rsID":          row["rsID"],
        "gene":          row["gene"],
        "chr":           row["chr"],
        "location":      row["location"],
        "ref":           row["ref"],
        "alt":           row["alt"],
        "kaviar_frq":    row["kaviar_frq"],
        "kaviar_number": int(row["kaviar_number"]),
        "kaviar_n":      kaviar_n,
        "KAZ_freq":      row["KAZ"],
        "EAS_freq":      row["EAS"],
        "EUR_freq":      row["EUR"],
    }

    # --- A) Each population vs Kaviar global (per-variant N) ---
    for pop in POPS:
        p, d = fisher_vs_reference(
            row[pop],          N_SAMPLES[pop],
            row["kaviar_frq"], kaviar_n
        )
        snp_result[f"{pop}_vs_global_p"]    = p
        snp_result[f"{pop}_vs_global_dir"]  = d
        snp_result[f"{pop}_vs_global_slog"] = signed_log10(p, d)

    # --- B) KAZ vs EUR ---
    p, d = fisher_vs_reference(
        row["KAZ"], N_SAMPLES["KAZ"],
        row["EUR"], N_SAMPLES["EUR"]
    )
    snp_result["KAZ_vs_EUR_p"]    = p
    snp_result["KAZ_vs_EUR_dir"]  = d
    snp_result["KAZ_vs_EUR_slog"] = signed_log10(p, d)

    # --- C) KAZ vs EAS ---
    p, d = fisher_vs_reference(
        row["KAZ"], N_SAMPLES["KAZ"],
        row["EAS"], N_SAMPLES["EAS"]
    )
    snp_result["KAZ_vs_EAS_p"]    = p
    snp_result["KAZ_vs_EAS_dir"]  = d
    snp_result["KAZ_vs_EAS_slog"] = signed_log10(p, d)

    # --- D) EUR vs EAS (for completeness / heatmap context) ---
    p, d = fisher_vs_reference(
        row["EUR"], N_SAMPLES["EUR"],
        row["EAS"], N_SAMPLES["EAS"]
    )
    snp_result["EUR_vs_EAS_p"]    = p
    snp_result["EUR_vs_EAS_dir"]  = d
    snp_result["EUR_vs_EAS_slog"] = signed_log10(p, d)

    results.append(snp_result)

res_df = pd.DataFrame(results)


# =============================================================================
# 5.  BONFERRONI CORRECTION
#     Applied per comparison column (same logic as paper: n_hypotheses = n_SNPs)
# =============================================================================

P_COLS = [
    "KAZ_vs_global_p", "EAS_vs_global_p", "EUR_vs_global_p",
    "KAZ_vs_EUR_p", "KAZ_vs_EAS_p", "EUR_vs_EAS_p"
]

for pcol in P_COLS:
    _, adj, _, _ = multipletests(res_df[pcol], alpha=ALPHA, method="bonferroni")
    res_df[pcol.replace("_p", "_adj_p")] = adj
    res_df[pcol.replace("_p", "_sig")]   = adj < ALPHA


# =============================================================================
# 6.  ENRICHMENT / DEPLETION STATUS LABELS
# =============================================================================

def status(row, comparison):
    if not row[f"{comparison}_sig"]:
        return "similar"
    return "enriched" if row[f"{comparison}_dir"] == 1 else "depleted"

for comp in ["KAZ_vs_global", "EAS_vs_global", "EUR_vs_global",
             "KAZ_vs_EUR", "KAZ_vs_EAS", "EUR_vs_EAS"]:
    res_df[f"{comp}_status"] = res_df.apply(lambda r: status(r, comp), axis=1)


# =============================================================================
# 7.  HEATMAP  — signed log10(p), populations vs global  (Shin Fig. 2 style)
#     Columns: KAZ, EAS, EUR (each vs global)
#     Rows: SNPs, hierarchically clustered
# =============================================================================

HEAT_COLS = {
    "KAZ": "KAZ_vs_global_slog",
    "EAS": "EAS_vs_global_slog",
    "EUR": "EUR_vs_global_slog",
}

heat_data = res_df[list(HEAT_COLS.values())].values
heat_df   = pd.DataFrame(heat_data,
                          index=res_df["rsID"],
                          columns=list(HEAT_COLS.keys()))

# Mask non-significant cells to 0 (as in Shin et al.)
sig_mask_global = np.column_stack([
    res_df["KAZ_vs_global_sig"].values,
    res_df["EAS_vs_global_sig"].values,
    res_df["EUR_vs_global_sig"].values,
])
heat_masked = heat_df.values.copy()
heat_masked[~sig_mask_global] = 0.0


def plot_heatmap(data, row_labels, col_labels, title, filename,
                 figsize_scale=(1.4, 0.30)):
    """Hierarchically clustered heatmap matching Shin et al. colour scheme."""
    # Cluster rows (SNPs); skip column clustering with only 3 cols
    if data.shape[0] > 1:
        row_link  = linkage(pdist(data, metric="euclidean"), method="average")
        row_order = leaves_list(row_link)
    else:
        row_order = [0]

    data_ord   = data[row_order, :]
    row_labels_ord = [row_labels[i] for i in row_order]

    vmax = np.percentile(np.abs(data[data != 0]), 95) if (data != 0).any() else 1
    vmax = max(vmax, 1)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shin_heatmap",
        [(0.35, 0.0,  0.55),   # deep purple  (depleted)
         (0.70, 0.50, 0.88),   # light purple
         (1.0,  1.0,  1.0),    # white         (neutral / non-sig)
         (1.0,  0.68, 0.58),   # light red
         (0.85, 0.10, 0.10)],  # deep red      (enriched)
        N=512
    )

    fig_w = max(5, len(col_labels) * figsize_scale[0])
    fig_h = max(6, len(row_labels) * figsize_scale[1])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(data_ord, aspect="auto", cmap=cmap,
                   vmin=-vmax, vmax=vmax, interpolation="nearest")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(row_labels_ord)))
    ax.set_yticklabels(row_labels_ord, fontsize=max(4, min(8, 200//len(row_labels_ord))))

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Signed log₁₀(p)\n(+ enriched, − depleted)", fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {filename}")


print("\n[1] Generating heatmaps...")

# Full heatmap (all values)
plot_heatmap(heat_df.values, list(res_df["rsID"]),
             list(HEAT_COLS.keys()),
             "Cardio SNPs — signed log₁₀(p) vs global frequency\n(KAZ, EAS, EUR)",
             "heatmap_vs_global_all.png")

# Significant only
plot_heatmap(heat_masked, list(res_df["rsID"]),
             list(HEAT_COLS.keys()),
             f"Cardio SNPs — significant enrichment/depletion vs global\n(Bonferroni p < {bonf_thresh:.2e})",
             "heatmap_vs_global_significant.png")

# Pairwise heatmap: KAZ vs EUR, KAZ vs EAS, EUR vs EAS
pair_data = res_df[["KAZ_vs_EUR_slog", "KAZ_vs_EAS_slog", "EUR_vs_EAS_slog"]].values
pair_labels = ["KAZ vs EUR", "KAZ vs EAS", "EUR vs EAS"]
pair_sig = np.column_stack([
    res_df["KAZ_vs_EUR_sig"].values,
    res_df["KAZ_vs_EAS_sig"].values,
    res_df["EUR_vs_EAS_sig"].values,
])
pair_masked = pair_data.copy()
pair_masked[~pair_sig] = 0.0

plot_heatmap(pair_masked, list(res_df["rsID"]),
             pair_labels,
             f"Cardio SNPs — pairwise comparisons\n(Bonferroni p < {bonf_thresh:.2e})",
             "heatmap_pairwise_significant.png")


# =============================================================================
# 8.  ENRICHMENT / DEPLETION COUNTS
# =============================================================================

print("\n[2] Enrichment / depletion counts vs global:")
print(f"{'Population':<12} {'Enriched':>10} {'Depleted':>10} {'Similar':>10}")
for pop in POPS:
    comp = f"{pop}_vs_global"
    enr = (res_df[f"{comp}_status"] == "enriched").sum()
    dep = (res_df[f"{comp}_status"] == "depleted").sum()
    sim = (res_df[f"{comp}_status"] == "similar").sum()
    print(f"{pop:<12} {enr:>10} {dep:>10} {sim:>10}")

print("\n[3] KAZ pairwise comparison counts:")
for comp, label in [("KAZ_vs_EUR", "KAZ vs EUR"), ("KAZ_vs_EAS", "KAZ vs EAS")]:
    enr = (res_df[f"{comp}_status"] == "enriched").sum()
    dep = (res_df[f"{comp}_status"] == "depleted").sum()
    sim = (res_df[f"{comp}_status"] == "similar").sum()
    print(f"  {label}: {enr} enriched, {dep} depleted, {sim} similar")


# =============================================================================
# 9.  SNPS THAT DIFFER IN KAZ vs BOTH EUR AND EAS
# =============================================================================

kaz_diff_both = res_df[
    res_df["KAZ_vs_EUR_sig"] & res_df["KAZ_vs_EAS_sig"]
].copy()

kaz_diff_eur_only = res_df[
    res_df["KAZ_vs_EUR_sig"] & ~res_df["KAZ_vs_EAS_sig"]
].copy()

kaz_diff_eas_only = res_df[
    ~res_df["KAZ_vs_EUR_sig"] & res_df["KAZ_vs_EAS_sig"]
].copy()

print(f"\n[4] KAZ-specific SNPs:")
print(f"  Different from BOTH EUR and EAS: {len(kaz_diff_both)}")
print(f"  Different from EUR only:         {len(kaz_diff_eur_only)}")
print(f"  Different from EAS only:         {len(kaz_diff_eas_only)}")

if len(kaz_diff_both) > 0:
    print("\n  SNPs different in KAZ vs both EUR and EAS:")
    display_cols = ["rsID", "gene", "KAZ_freq", "EUR_freq", "EAS_freq",
                    "kaviar_frq", "KAZ_vs_EUR_status", "KAZ_vs_EAS_status",
                    "KAZ_vs_EUR_adj_p", "KAZ_vs_EAS_adj_p"]
    print(kaz_diff_both[display_cols].to_string(index=False))


# =============================================================================
# 10. COMPOSITE GRS
#     GRS = sum(2 × EAF_pop) / (2 × I)
#     No OR weights here since this is a frequency-only input file.
#     If you have OR/beta values, add them as a column and uncomment PRS block.
# =============================================================================

print("\n[5] Genetic Risk Scores:")

grs = {}
for pop in ["KAZ", "EAS", "EUR"]:
    xi  = 2 * df[pop].values
    grs[pop] = xi.sum() / (2 * N_SNPS)

grs_df = pd.DataFrame({"Population": list(grs.keys()),
                        "GRS": list(grs.values())})
print(grs_df.to_string(index=False))
print(f"  (Random expectation = 0.5; scores reflect mean effect allele dosage)")


# =============================================================================
# 11. GRS BAR CHART
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
pops_plot = ["KAZ", "EAS", "EUR"]
grs_vals  = [grs[p] for p in pops_plot]
colors    = ["#C0392B", "#27AE60", "#2980B9"]
bars = ax.bar(pops_plot, grs_vals, color=colors, alpha=0.85, width=0.5)
ax.axhline(0.5, color="gray", lw=1.5, ls="--", label="Random expectation (0.5)")
for bar, val in zip(bars, grs_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Composite GRS", fontsize=11)
ax.set_title("Genetic Risk Score\nCardiovascular SNPs", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, max(grs_vals) * 1.2)

# Allele frequency comparison scatter for top KAZ-specific SNPs
ax2 = axes[1]
if len(kaz_diff_both) > 0:
    plot_snps = kaz_diff_both.head(20)
else:
    plot_snps = res_df[res_df["KAZ_vs_EUR_sig"] | res_df["KAZ_vs_EAS_sig"]].head(20)

x = np.arange(len(plot_snps))
w = 0.25
ax2.bar(x - w,   plot_snps["KAZ_freq"], w, label="KAZ", color="#C0392B", alpha=0.85)
ax2.bar(x,       plot_snps["EAS_freq"], w, label="EAS", color="#27AE60", alpha=0.85)
ax2.bar(x + w,   plot_snps["EUR_freq"], w, label="EUR", color="#2980B9", alpha=0.85)
ax2.set_xticks(x)
ax2.set_xticklabels(plot_snps["rsID"], rotation=90, fontsize=7)
ax2.set_ylabel("Allele Frequency", fontsize=11)
title_str = "KAZ vs EUR & EAS" if len(kaz_diff_both) > 0 else "KAZ significantly different SNPs"
ax2.set_title(f"Top SNPs — {title_str}", fontsize=11)
ax2.legend(fontsize=9)

plt.tight_layout()
plt.savefig("risk_scores_and_snps.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n  Saved: risk_scores_and_snps.png")


# =============================================================================
# 12. SAVE RESULTS
# =============================================================================

# Full results table
res_df.to_csv("kaz_snp_analysis_full.csv", index=False)

# KAZ-specific SNPs only
kaz_sig = res_df[res_df["KAZ_vs_EUR_sig"] | res_df["KAZ_vs_EAS_sig"]].copy()
kaz_sig["differs_from"] = kaz_sig.apply(
    lambda r: ("EUR+EAS" if r["KAZ_vs_EUR_sig"] and r["KAZ_vs_EAS_sig"]
               else "EUR" if r["KAZ_vs_EUR_sig"]
               else "EAS"), axis=1
)
kaz_sig.to_csv("kaz_specific_snps.csv", index=False)

print(f"\n[6] Output files:")
print(f"  kaz_snp_analysis_full.csv     — all SNPs, all p-values and statuses")
print(f"  kaz_specific_snps.csv         — SNPs where KAZ differs significantly")
print(f"  heatmap_vs_global_all.png     — full heatmap vs global")
print(f"  heatmap_vs_global_significant.png — significant only vs global")
print(f"  heatmap_pairwise_significant.png  — KAZ vs EUR/EAS pairwise")
print(f"  risk_scores_and_snps.png      — GRS bars + allele freq comparison")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"  Total SNPs:               {N_SNPS}")
print(f"  Bonferroni threshold:     p < {bonf_thresh:.2e}")
print(f"  KAZ differs from EUR:     {res_df['KAZ_vs_EUR_sig'].sum()}")
print(f"  KAZ differs from EAS:     {res_df['KAZ_vs_EAS_sig'].sum()}")
print(f"  KAZ differs from BOTH:    {len(kaz_diff_both)}")
print(f"\n  Global reference: Kaviar (per-variant AN from kaviar_number column)")
print(f"  KAZ N={N_SAMPLES['KAZ']}, EAS N={N_SAMPLES['EAS']}, EUR N={N_SAMPLES['EUR']}")
