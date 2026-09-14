"""
plot_rd_curves.py

Lê o arquivo benchmark_all.tsv e gera uma curva Rate-Distortion (RD)
comparando os modos 'standard' e 'erp'.

- Eixo X: PSNR (dB)
- Eixo Y: Rate (bpp)

A curva é a MÉDIA sobre todas as imagens para cada (modo, lambda).
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def bd_rate(R1, PSNR1, R2, PSNR2):
    """
    Computes Bjontegaard Delta Rate (BD-Rate)
    R1, PSNR1: reference (baseline)
    R2, PSNR2: test
    Negative BD-Rate means test is better (bitrate savings).
    """
    lR1 = np.log10(R1)
    lR2 = np.log10(R2)
    
    # Polynomial degree (max 3, but lower if fewer points)
    deg1 = min(3, len(PSNR1) - 1)
    deg2 = min(3, len(PSNR2) - 1)
    
    p1 = np.polyfit(PSNR1, lR1, deg1)
    p2 = np.polyfit(PSNR2, lR2, deg2)

    # Integration interval
    min_int = max(min(PSNR1), min(PSNR2))
    max_int = min(max(PSNR1), max(PSNR2))

    # Calculate integral
    int1 = np.polyint(p1)
    int2 = np.polyint(p2)
    
    avg_diff = (np.polyval(int2, max_int) - np.polyval(int2, min_int)) - \
               (np.polyval(int1, max_int) - np.polyval(int1, min_int))
    
    avg_diff = avg_diff / (max_int - min_int)
    
    return (10 ** avg_diff - 1) * 100

def bd_psnr(R1, PSNR1, R2, PSNR2):
    """
    Computes Bjontegaard Delta PSNR (BD-PSNR)
    R1, PSNR1: reference (baseline)
    R2, PSNR2: test
    Positive BD-PSNR means test is better (higher PSNR).
    """
    lR1 = np.log10(R1)
    lR2 = np.log10(R2)
    
    # Polynomial degree (max 3, but lower if fewer points)
    deg1 = min(3, len(PSNR1) - 1)
    deg2 = min(3, len(PSNR2) - 1)
    
    p1 = np.polyfit(lR1, PSNR1, deg1)
    p2 = np.polyfit(lR2, PSNR2, deg2)

    # Integration interval
    min_int = max(min(lR1), min(lR2))
    max_int = min(max(lR1), max(lR2))

    # Calculate integral
    int1 = np.polyint(p1)
    int2 = np.polyint(p2)
    
    avg_diff = (np.polyval(int2, max_int) - np.polyval(int2, min_int)) - \
               (np.polyval(int1, max_int) - np.polyval(int1, min_int))
    
    avg_diff = avg_diff / (max_int - min_int)
    
    return avg_diff

# ── configuração ──────────────────────────────────────────────────────────────
TSV_PATH = os.path.join(os.path.dirname(__file__),
                        "benchmark_results_all_scenarios_10_imgs_3_lmbdas", "benchmark_all.tsv")
OUT_PATH  = os.path.join(os.path.dirname(__file__),
                        "benchmark_results_all_scenarios_10_imgs_3_lmbdas", "rd_curve.png")

# ── estilo ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#161b27",
    "axes.edgecolor":    "#2e3650",
    "axes.labelcolor":   "#c8d0e8",
    "axes.titlecolor":   "#e8eaf6",
    "xtick.color":       "#8892b0",
    "ytick.color":       "#8892b0",
    "grid.color":        "#1e2640",
    "grid.linewidth":    0.8,
    "text.color":        "#c8d0e8",
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

PALETTE = {
    "standard":      {"color": "#64b5f6", "marker": "o", "ls": "-",  "lw": 2.2},
    "erp":           {"color": "#f06292", "marker": "s", "ls": "--", "lw": 2.2},
    "erp_gw":        {"color": "#ba68c8", "marker": "D", "ls": "-.", "lw": 2.2},
    "ws_mse":        {"color": "#4db6ac", "marker": "v", "ls": "-",  "lw": 2.2},
    "ws_mse_erp":    {"color": "#ffb74d", "marker": "^", "ls": "--", "lw": 2.2},
    "ws_mse_erp_gw": {"color": "#e57373", "marker": "p", "ls": "-.", "lw": 2.2},
}
LABEL = {
    "standard":      "Standard (MSE)",
    "erp":           "ERP (MSE)",
    "erp_gw":        "ERP + Gauss Weights (MSE)",
    "ws_mse":        "Standard (WS-MSE)",
    "ws_mse_erp":    "ERP (WS-MSE)",
    "ws_mse_erp_gw": "ERP + Gauss Weights (WS-MSE)",
}

# ── leitura ───────────────────────────────────────────────────────────────────
df = pd.read_csv(TSV_PATH, sep=r"\s+", engine="python")
df.columns = df.columns.str.strip()
df["ws_psnr_db"]  = df["ws_psnr_db"].astype(float)
df["rate_bpp"] = df["rate_bpp"].astype(float)
df["lmbda"]    = df["lmbda"].astype(float)
df["mode"]     = df["mode"].str.strip()

# ── média sobre imagens para cada (modo, lambda) ──────────────────────────────
avg = (df
       .groupby(["mode", "lmbda"], as_index=False)
       .agg(ws_psnr_mean=("ws_psnr_db", "mean"),
            psnr_mean=("psnr_db", "mean"),
            bpp_mean=("rate_bpp", "mean"))
       .sort_values("ws_psnr_mean"))

# ── BD-Rate Calculation ───────────────────────────────────────────────────────
bd_rate_path = os.path.join(os.path.dirname(TSV_PATH), "bd_rate.txt")
bd_psnr_path = os.path.join(os.path.dirname(TSV_PATH), "bd_psnr.txt")
bd_rate_lines = []
bd_psnr_lines = []

def compute_bd_metrics(metric_col, title):
    bd_rate_lines.append(f"=== BD-Rate Analysis ({title} | Baseline: standard) ===")
    bd_psnr_lines.append(f"=== BD-PSNR Analysis ({title} | Baseline: standard) ===")
    print("\n" + bd_rate_lines[-1])
    print(bd_psnr_lines[-1])
    
    standard_baseline = avg[avg["mode"] == "standard"].sort_values(metric_col)
    if not standard_baseline.empty and len(standard_baseline) > 2:
        R1 = standard_baseline["bpp_mean"].values
        PSNR1 = standard_baseline[metric_col].values
        for mode in avg["mode"].unique():
            if mode == "standard":
                continue
            test_curve = avg[avg["mode"] == mode].sort_values(metric_col)
            if len(test_curve) > 2:
                R2 = test_curve["bpp_mean"].values
                PSNR2 = test_curve[metric_col].values
                
                # Rate
                try:
                    bd_r = bd_rate(R1, PSNR1, R2, PSNR2)
                    line_r = f"BD-Rate vs {mode:15s}: {bd_r:7.2f}%"
                except Exception as e:
                    line_r = f"BD-Rate vs {mode:15s}: N/A (Error: {e})"
                
                # PSNR
                try:
                    bd_p = bd_psnr(R1, PSNR1, R2, PSNR2)
                    line_p = f"BD-PSNR vs {mode:15s}: {bd_p:7.3f} dB"
                except Exception as e:
                    line_p = f"BD-PSNR vs {mode:15s}: N/A (Error: {e})"
            else:
                line_r = f"BD-Rate vs {mode:15s}: N/A (Not enough points)"
                line_p = f"BD-PSNR vs {mode:15s}: N/A (Not enough points)"
                
            print(line_r)
            print(line_p)
            bd_rate_lines.append(line_r)
            bd_psnr_lines.append(line_p)
            
    footer = "============================================="
    print(footer + "\n")
    bd_rate_lines.append(footer)
    bd_psnr_lines.append(footer)

compute_bd_metrics("ws_psnr_mean", "WS-PSNR")
compute_bd_metrics("psnr_mean", "Standard PSNR")

with open(bd_rate_path, "w") as f:
    f.write("\n".join(bd_rate_lines) + "\n")
with open(bd_psnr_path, "w") as f:
    f.write("\n".join(bd_psnr_lines) + "\n")
print(f"BD-Rates salvos em: {bd_rate_path}")
print(f"BD-PSNRs salvos em: {bd_psnr_path}")

# ── plot ───────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))

for mode, grp in avg.groupby("mode"):
    grp = grp.sort_values("ws_psnr_mean")
    s = PALETTE[mode]
    ax.plot(grp["ws_psnr_mean"], grp["bpp_mean"],
            color=s["color"], marker=s["marker"], ls=s["ls"], lw=s["lw"],
            markersize=8, markeredgewidth=1.4, markeredgecolor="#0f1117",
            label=LABEL[mode], zorder=3)

    # anotar cada ponto com o valor de λ
    for _, row in grp.iterrows():
        lbl = f"λ={row['lmbda']:.4g}"
        offset_x = 0.08
        offset_y = 0.012
        if "erp" in mode:
            offset_y = -0.022
        ax.annotate(lbl,
                    xy=(row["ws_psnr_mean"], row["bpp_mean"]),
                    xytext=(row["ws_psnr_mean"] + offset_x, row["bpp_mean"] + offset_y),
                    fontsize=7.5, color=s["color"], alpha=0.85,
                    arrowprops=None)

ax.set_xlabel("WS-PSNR (dB)", fontsize=13, labelpad=8)
ax.set_ylabel("Rate (bpp)", fontsize=13, labelpad=8)
ax.set_title("Rate-Distortion: Standard vs ERP Scenarios\n(média sobre todas as imagens)",
             fontsize=14, pad=14, fontweight="bold")

ax.grid(True, which="both", alpha=0.4)
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

# legenda com fundo semi-transparente
leg = ax.legend(fontsize=11, framealpha=0.25, facecolor="#1e2640",
                edgecolor="#2e3650", loc="upper left")
for text in leg.get_texts():
    text.set_color("#e8eaf6")

# A área preenchida foi removida pois 6 curvas tornariam o gráfico ilegível

# bordas do plot
for spine in ax.spines.values():
    spine.set_edgecolor("#2e3650")

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Plot salvo em: {OUT_PATH}")
plt.show()
