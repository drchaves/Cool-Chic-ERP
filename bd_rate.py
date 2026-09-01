"""
bd_rate.py

Calcula a métrica Bjontegaard Delta-Rate (BD-Rate) e BD-PSNR
comparando os modos 'standard' (baseline) e 'erp' (proposto).

BD-Rate < 0  →  ERP precisa de MENOS bits para a mesma qualidade (melhor)
BD-PSNR > 0  →  ERP tem MAIS qualidade para o mesmo bitrate   (melhor)

Referência: Bjontegaard, G. (2001). "Calculation of Average PSNR Differences
between RD Curves." VCEG-M33.
"""

import os
import numpy as np
import pandas as pd

# ── caminho do TSV ────────────────────────────────────────────────────────────
TSV_PATH = os.path.join(os.path.dirname(__file__),
                        "benchmark_results_ctc_poles", "benchmark_all.tsv")


# ─────────────────────────────────────────────────────────────────────────────
# Implementação Bjontegaard (cúbico, integração analítica)
# ─────────────────────────────────────────────────────────────────────────────

def bd_rate(psnr1, rate1, psnr2, rate2):
    """
    Calcula BD-Rate entre a curva de referência (1) e a curva proposta (2).

    Parâmetros
    ----------
    psnr1, rate1 : arrays da curva de REFERÊNCIA  (standard)
    psnr2, rate2 : arrays da curva PROPOSTA        (erp)

    Retorna
    -------
    bd_rate_pct : float   — diferença percentual média de bitrate
                            (negativo = proposto é melhor)
    bd_psnr_db  : float   — diferença média de PSNR em dB
                            (positivo = proposto é melhor)
    """
    # trabalha em log(rate) para linearizar a curva RD
    log_r1 = np.log(np.asarray(rate1, dtype=float))
    log_r2 = np.log(np.asarray(rate2, dtype=float))
    p1     = np.asarray(psnr1, dtype=float)
    p2     = np.asarray(psnr2, dtype=float)

    # ordena por PSNR crescente
    idx1 = np.argsort(p1); p1 = p1[idx1]; log_r1 = log_r1[idx1]
    idx2 = np.argsort(p2); p2 = p2[idx2]; log_r2 = log_r2[idx2]

    # ajuste polinomial cúbico (grau 3)
    coef1 = np.polyfit(p1, log_r1, 3)
    coef2 = np.polyfit(p2, log_r2, 3)

    # intervalo de integração = intersecção dos intervalos de PSNR
    psnr_lo = max(p1.min(), p2.min())
    psnr_hi = min(p1.max(), p2.max())

    if psnr_lo >= psnr_hi:
        return float("nan"), float("nan")

    # integral analítica do polinômio de grau 3  →  polinômio de grau 4
    int1 = np.polyint(coef1)
    int2 = np.polyint(coef2)

    # valor médio de log(rate) no intervalo
    def poly_avg(coef_int, lo, hi):
        return (np.polyval(coef_int, hi) - np.polyval(coef_int, lo)) / (hi - lo)

    avg_log_r1 = poly_avg(int1, psnr_lo, psnr_hi)
    avg_log_r2 = poly_avg(int2, psnr_lo, psnr_hi)

    bd_rate_pct = (np.exp(avg_log_r2 - avg_log_r1) - 1) * 100.0

    # BD-PSNR: ajusta PSNR em função de log(rate) e integra
    coef_p1 = np.polyfit(log_r1, p1, 3)
    coef_p2 = np.polyfit(log_r2, p2, 3)

    log_lo = max(log_r1.min(), log_r2.min())
    log_hi = min(log_r1.max(), log_r2.max())

    if log_lo >= log_hi:
        bd_psnr_db = float("nan")
    else:
        int_p1 = np.polyint(coef_p1)
        int_p2 = np.polyint(coef_p2)
        avg_p1 = poly_avg(int_p1, log_lo, log_hi)
        avg_p2 = poly_avg(int_p2, log_lo, log_hi)
        bd_psnr_db = avg_p2 - avg_p1

    return bd_rate_pct, bd_psnr_db


# ─────────────────────────────────────────────────────────────────────────────
# Leitura e processamento
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(TSV_PATH, sep=r"\s+", engine="python")
df.columns    = df.columns.str.strip()
df["mode"]     = df["mode"].str.strip()
df["psnr_db"]  = df["psnr_db"].astype(float)
df["rate_bpp"] = df["rate_bpp"].astype(float)
df["lmbda"]    = df["lmbda"].astype(float)

images = sorted(df["image_name"].unique())

# ── BD-Rate por imagem ────────────────────────────────────────────────────────
rows = []
for img in images:
    sub = df[df["image_name"] == img]
    std = sub[sub["mode"] == "standard"].sort_values("lmbda", ascending=False)
    erp = sub[sub["mode"] == "erp"].sort_values("lmbda", ascending=False)

    if len(std) < 4 or len(erp) < 4:
        print(f"  [AVISO] {img}: pontos insuficientes — pulando.")
        continue

    bdr, bdp = bd_rate(std["psnr_db"], std["rate_bpp"],
                       erp["psnr_db"], erp["rate_bpp"])
    rows.append({"image": img, "bd_rate_%": bdr, "bd_psnr_dB": bdp})

per_img = pd.DataFrame(rows)

# ── BD-Rate global (média sobre imagens) ─────────────────────────────────────
avg_df = (df.groupby(["mode", "lmbda"], as_index=False)
            .agg(psnr_db=("psnr_db", "mean"), rate_bpp=("rate_bpp", "mean"))
            .sort_values("lmbda", ascending=False))

std_avg = avg_df[avg_df["mode"] == "standard"]
erp_avg = avg_df[avg_df["mode"] == "erp"]

global_bdr, global_bdp = bd_rate(
    std_avg["psnr_db"], std_avg["rate_bpp"],
    erp_avg["psnr_db"], erp_avg["rate_bpp"])

# ─────────────────────────────────────────────────────────────────────────────
# Relatório
# ─────────────────────────────────────────────────────────────────────────────

SEP = "═" * 68

print(f"\n{SEP}")
print("  Bjontegaard Delta-Rate (BD-Rate)  —  ERP vs Standard")
print(f"{SEP}")
print(f"  Referência (baseline) : Standard (ARM retangular)")
print(f"  Proposto              : ERP (ARM geodésico 360°)")
print(f"  BD-Rate < 0  →  ERP precisa de MENOS bits (melhor)")
print(f"  BD-PSNR > 0  →  ERP tem MAIS qualidade    (melhor)")
print(f"{SEP}\n")

# Por imagem
print(f"  {'Imagem':<48}  {'BD-Rate':>8}  {'BD-PSNR':>9}")
print(f"  {'─'*48}  {'─'*8}  {'─'*9}")
for _, r in per_img.iterrows():
    sign_r = "✅" if r["bd_rate_%"]  < 0 else "⚠️ "
    sign_p = "✅" if r["bd_psnr_dB"] > 0 else "⚠️ "
    print(f"  {r['image']:<48}  "
          f"{sign_r} {r['bd_rate_%']:+6.2f}%  "
          f"{sign_p} {r['bd_psnr_dB']:+6.4f} dB")

print(f"\n  {'─'*48}  {'─'*8}  {'─'*9}")
mean_bdr = per_img["bd_rate_%"].mean()
mean_bdp = per_img["bd_psnr_dB"].mean()
sign_r = "✅" if mean_bdr  < 0 else "⚠️ "
sign_p = "✅" if mean_bdp  > 0 else "⚠️ "
print(f"  {'MÉDIA por imagem':<48}  "
      f"{sign_r} {mean_bdr:+6.2f}%  "
      f"{sign_p} {mean_bdp:+6.4f} dB")
print(f"  {'GLOBAL (curva média)':<48}  "
      f"{'✅' if global_bdr < 0 else '⚠️ '} {global_bdr:+6.2f}%  "
      f"{'✅' if global_bdp > 0 else '⚠️ '} {global_bdp:+6.4f} dB")
print(f"\n{SEP}\n")

# Salva CSV
out_csv = os.path.join(os.path.dirname(TSV_PATH), "bd_rate_results.csv")
per_img.to_csv(out_csv, index=False, float_format="%.4f")
print(f"  Resultados por imagem salvos em: {out_csv}\n")
