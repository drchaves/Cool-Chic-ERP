"""
bd_rate.py

Calcula a métrica Bjontegaard Delta-Rate (BD-Rate) e BD-PSNR comparando
todos os modos não-standard presentes no TSV contra o modo 'standard' (baseline).

Detecta automaticamente os modos disponíveis no TSV — não é necessário
editar o script quando novos experimentos (ex: erp_polar30, erp_polar60) são adicionados.

BD-Rate < 0  →  proposto precisa de MENOS bits para a mesma qualidade (melhor)
BD-PSNR > 0  →  proposto tem MAIS qualidade para o mesmo bitrate   (melhor)

Referência: Bjontegaard, G. (2001). "Calculation of Average PSNR Differences
between RD Curves." VCEG-M33.

Uso:
    python bd_rate.py                              # usa TSV padrão
    python bd_rate.py --tsv path/to/benchmark.tsv  # TSV customizado
    python bd_rate.py --mode erp_polar30           # compara só esse modo
    python bd_rate.py --all-dirs                   # processa todos os benchmark_results_* encontrados
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── TSV padrão ────────────────────────────────────────────────────────────────
DEFAULT_TSV = os.path.join(os.path.dirname(__file__),
                           "benchmark_results_ctc_poles", "benchmark_all.tsv")

BASELINE_MODE = "standard"


# ─────────────────────────────────────────────────────────────────────────────
# Implementação Bjontegaard (cúbico, integração analítica)
# ─────────────────────────────────────────────────────────────────────────────

def bd_rate(psnr1, rate1, psnr2, rate2):
    """
    Calcula BD-Rate entre a curva de referência (1) e a curva proposta (2).

    Parâmetros
    ----------
    psnr1, rate1 : arrays da curva de REFERÊNCIA  (standard)
    psnr2, rate2 : arrays da curva PROPOSTA        (erp / erp_polar*)

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
# Processamento de um TSV
# ─────────────────────────────────────────────────────────────────────────────

def process_tsv(tsv_path: str, filter_mode: str = None, save_csv: bool = True) -> dict:
    """Lê um TSV de benchmark, detecta os modos e calcula BD-Rate de cada um.

    Args:
        tsv_path:    Caminho para o arquivo TSV.
        filter_mode: Se não-None, processa apenas este modo.
        save_csv:    Se True, salva um CSV por modo em <dir>/<modo>_bd_rate.csv

    Returns:
        dict: {modo: DataFrame com colunas [image, bd_rate_%, bd_psnr_dB]}
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        print(f"  [ERRO] TSV não encontrado: {tsv_path}", file=sys.stderr)
        return {}

    df = pd.read_csv(tsv_path, sep=r"\s+", engine="python")
    df.columns     = df.columns.str.strip()
    df["mode"]     = df["mode"].str.strip()
    df["psnr_db"]  = df["psnr_db"].astype(float)
    df["rate_bpp"] = df["rate_bpp"].astype(float)
    df["lmbda"]    = df["lmbda"].astype(float)

    all_modes = sorted(df["mode"].unique())
    if BASELINE_MODE not in all_modes:
        print(f"  [ERRO] Modo baseline '{BASELINE_MODE}' não encontrado em {tsv_path.name}",
              file=sys.stderr)
        print(f"  Modos disponíveis: {all_modes}", file=sys.stderr)
        return {}

    # Modos a comparar (todos menos o baseline)
    candidate_modes = [m for m in all_modes if m != BASELINE_MODE]
    if filter_mode:
        if filter_mode not in candidate_modes:
            print(f"  [ERRO] Modo '{filter_mode}' não encontrado. Disponíveis: {candidate_modes}",
                  file=sys.stderr)
            return {}
        candidate_modes = [filter_mode]

    images = sorted(df["image_name"].unique())
    results = {}

    SEP = "═" * 72

    print(f"\n{SEP}")
    print(f"  TSV: {tsv_path.name}")
    print(f"  Modos detectados: {all_modes}")
    print(f"  Baseline: {BASELINE_MODE!r}  |  Comparando: {candidate_modes}")
    print(f"{SEP}")

    for mode in candidate_modes:
        print(f"\n  ── Modo: {mode!r} vs {BASELINE_MODE!r} ──")
        print(f"  {'Imagem':<50}  {'BD-Rate':>8}  {'BD-PSNR':>9}")
        print(f"  {'─'*50}  {'─'*8}  {'─'*9}")

        rows = []
        for img in images:
            sub  = df[df["image_name"] == img]
            std  = sub[sub["mode"] == BASELINE_MODE].sort_values("lmbda", ascending=False)
            prop = sub[sub["mode"] == mode].sort_values("lmbda", ascending=False)

            if len(std) < 4 or len(prop) < 4:
                print(f"  [AVISO] {img}: pontos insuficientes ({len(std)} std, {len(prop)} {mode}) — pulando.")
                continue

            bdr, bdp = bd_rate(std["psnr_db"], std["rate_bpp"],
                               prop["psnr_db"], prop["rate_bpp"])

            flag_r = "✅" if bdr < 0 else "⚠️ "
            flag_p = "✅" if bdp > 0 else "⚠️ "
            print(f"  {img:<50}  {flag_r} {bdr:+6.2f}%  {flag_p} {bdp:+6.4f} dB")
            rows.append({"image": img, "bd_rate_%": round(bdr, 4), "bd_psnr_dB": round(bdp, 4)})

        if not rows:
            print(f"  [AVISO] Nenhuma imagem calculada para o modo {mode!r}.")
            continue

        mode_df = pd.DataFrame(rows)
        mean_bdr = mode_df["bd_rate_%"].mean()
        mean_bdp = mode_df["bd_psnr_dB"].mean()
        print(f"  {'─'*50}  {'─'*8}  {'─'*9}")
        print(f"  {'MÉDIA':<50}  "
              f"{'✅' if mean_bdr<0 else '⚠️ '} {mean_bdr:+6.2f}%  "
              f"{'✅' if mean_bdp>0 else '⚠️ '} {mean_bdp:+6.4f} dB")

        results[mode] = mode_df

        if save_csv:
            # Nome do CSV inclui o modo para evitar sobrescrever outros experimentos
            out_csv = tsv_path.parent / f"bd_rate_{mode}.csv"
            mode_df.to_csv(out_csv, index=False, float_format="%.4f")
            print(f"\n  💾 Salvo: {out_csv}")

    # ── Sumário comparativo (se múltiplos modos) ──────────────────────────────
    if len(results) > 1:
        print(f"\n{SEP}")
        print(f"  SUMÁRIO COMPARATIVO  ({tsv_path.name})")
        print(f"  {'Modo':<25}  {'BD-Rate médio':>14}  {'BD-PSNR médio':>14}  {'Melhor que std?':>16}")
        print(f"  {'─'*25}  {'─'*14}  {'─'*14}  {'─'*16}")
        for mode, mode_df in results.items():
            m_bdr = mode_df["bd_rate_%"].mean()
            m_bdp = mode_df["bd_psnr_dB"].mean()
            better = "✅ SIM" if m_bdr < 0 else "❌ NÃO"
            print(f"  {mode:<25}  {m_bdr:>+12.2f}%  {m_bdp:>+12.4f} dB  {better:>16}")
        print(SEP)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "--tsv", type=str, default=DEFAULT_TSV,
        help=f"Caminho para o arquivo TSV de benchmark.\n"
             f"Padrão: {DEFAULT_TSV}",
    )
    p.add_argument(
        "--mode", type=str, default=None,
        help="Processar apenas este modo (ex: erp_polar30). "
             "Padrão: todos os modos não-standard.",
    )
    p.add_argument(
        "--no-csv", action="store_true",
        help="Não salvar CSVs de saída.",
    )
    p.add_argument(
        "--all-dirs", action="store_true",
        help="Processar todos os diretórios benchmark_results_* encontrados "
             "ao lado do script.",
    )
    return p.parse_args()


def find_all_benchmark_tsvs() -> list[Path]:
    """Encontra todos os benchmark_all.tsv nos diretórios benchmark_results_*."""
    root = Path(__file__).parent
    return sorted(root.glob("benchmark_results*/benchmark_all.tsv"))


def main():
    args = parse_args()

    if args.all_dirs:
        tsvs = find_all_benchmark_tsvs()
        if not tsvs:
            print("  Nenhum diretório benchmark_results_* encontrado.", file=sys.stderr)
            sys.exit(1)
        print(f"  Encontrados {len(tsvs)} TSV(s):")
        for t in tsvs:
            print(f"    {t}")
        for tsv in tsvs:
            process_tsv(str(tsv), filter_mode=args.mode, save_csv=not args.no_csv)
    else:
        process_tsv(args.tsv, filter_mode=args.mode, save_csv=not args.no_csv)


if __name__ == "__main__":
    main()
