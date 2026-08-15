#!/usr/bin/env bash
# compare_erp.sh
#
# Encodes the same image twice:
#   1. standard Cool-Chic ARM (rectangular causal mask)
#   2. ERP geodesic ARM (geodesic nearest-neighbour context)
# then prints a side-by-side comparison of PSNR and bpp.
#
# Usage:
#   bash compare_erp.sh                         # default image + lmbda
#   bash compare_erp.sh image.png 1e-3          # custom image and lambda
#
# Set FULL=1 to run without --debug (much slower, production quality):
#   FULL=1 bash compare_erp.sh

set -euo pipefail

IMAGE="${1:-/Users/diego/Documents/Faculdade/Pesquisa/Cool-Chic/samples/images/othim01.png}"
LMBDA="${2:-1e-3}"
#FULL="${FULL:-0}"
FULL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

WORKDIR_STD="${SCRIPT_DIR}/compare_erp/standard"
WORKDIR_ERP="${SCRIPT_DIR}/compare_erp/erp"
mkdir -p "${WORKDIR_STD}" "${WORKDIR_ERP}"

if [[ "${FULL}" == "1" ]]; then
    EXTRA_FLAGS=""
    echo "Full training mode (no --debug). This may take several minutes."
else
    EXTRA_FLAGS="--debug"
    echo "Debug mode (--debug). Very fast but low quality. Set FULL=1 for full training."
fi

echo "──────────────────────────────────────────────"
echo "Image  : ${IMAGE}"
echo "Lambda : ${LMBDA}"
echo "──────────────────────────────────────────────"

echo ""
echo "1/2 — Standard ARM (rectangular context)"
echo ""
"${PYTHON}" "${SCRIPT_DIR}/cc_encode.py" \
    --input   "${IMAGE}" \
    --output  "${WORKDIR_STD}/bitstream.cool" \
    --workdir "${WORKDIR_STD}" \
    --lmbda   "${LMBDA}" \
    ${EXTRA_FLAGS}

echo ""
echo "2/2 — ERP geodesic ARM"
echo ""
"${PYTHON}" "${SCRIPT_DIR}/cc_encode.py" \
    --input   "${IMAGE}" \
    --output  "${WORKDIR_ERP}/bitstream.cool" \
    --workdir "${WORKDIR_ERP}" \
    --lmbda   "${LMBDA}" \
    --erp_residue \
    ${EXTRA_FLAGS}

echo ""
echo "══════════════════════════════════════════════"
echo "  RESULTS COMPARISON"
echo "══════════════════════════════════════════════"

TSV_STD=$(find "${WORKDIR_STD}" -name "results_decoder.tsv" | head -1)
TSV_ERP=$(find "${WORKDIR_ERP}" -name "results_decoder.tsv" | head -1)

read_field() {
    local tsv="$1"
    local field="$2"
    awk -v f="${field}" '
        NR==1 { for(i=1;i<=NF;i++) if($i==f) col=i }
        NR==2 { print $col }
    ' "${tsv}"
}

PSNR_STD=$(read_field "${TSV_STD}" "psnr_db")
BPP_STD=$(read_field  "${TSV_STD}" "rate_bpp")
LOSS_STD=$(read_field "${TSV_STD}" "loss")
PSNR_ERP=$(read_field "${TSV_ERP}" "psnr_db")
BPP_ERP=$(read_field  "${TSV_ERP}" "rate_bpp")
LOSS_ERP=$(read_field "${TSV_ERP}" "loss")

DELTA_PSNR=$(awk "BEGIN {printf \"%.6f\", ${PSNR_ERP} - ${PSNR_STD}}")
DELTA_BPP=$(awk  "BEGIN {printf \"%.6f\", ${BPP_ERP}  - ${BPP_STD}}")

printf "\n%-28s  %-14s  %-14s\n" "Metric"          "Standard"      "ERP geodesic"
printf "%-28s  %-14s  %-14s\n"   "────────────────" "─────────────" "─────────────"
printf "%-28s  %-14s  %-14s\n"   "PSNR (dB)"       "${PSNR_STD}"   "${PSNR_ERP}"
printf "%-28s  %-14s  %-14s\n"   "Rate (bpp)"       "${BPP_STD}"    "${BPP_ERP}"
printf "%-28s  %-14s  %-14s\n"   "Loss (×1000)"     "${LOSS_STD}"   "${LOSS_ERP}"
printf "\n%-28s  %s dB\n"  "ΔPSNR (ERP − Std)"  "${DELTA_PSNR}"
printf "%-28s  %s bpp\n\n" "Δbpp  (ERP − Std)"  "${DELTA_BPP}"

BETTER=$(awk "BEGIN { print (${DELTA_PSNR}+0 > 0 || ${DELTA_BPP}+0 < 0) ? \"yes\" : \"no\" }")
if [[ "${BETTER}" == "yes" ]]; then
    echo "✅  ERP geodesic context improved the result."
else
    echo "⚠️   ERP geodesic context did NOT improve the result for this run."
fi

echo ""
echo "Results in:"
echo "  ${WORKDIR_STD}/results_decoder.tsv"
echo "  ${WORKDIR_ERP}/results_decoder.tsv"
