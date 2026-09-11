#!/usr/bin/env bash
# run_benchmark.sh
#
# Runs all combinations of:
#   - Images  : every *.png / *.yuv found in IMAGE_DIR
#   - Modes   : 6 scenarios (standard / erp / erp+gw / ws_mse / ws_mse+erp / ws_mse+erp+gw)
#   - Lambdas : values defined in LAMBDAS array
#
# Results are aggregated into a single TSV file so that all scenarios
# can later be loaded and compared in one place.
#
# Usage:
#   bash run_benchmark.sh [IMAGE_DIR] [OUTPUT_DIR]
#
# Defaults:
#   IMAGE_DIR  = ./samples/images
#   OUTPUT_DIR = ./benchmark_results
#
# Environment variables:
#   FULL=0   runs in --debug mode (fast, low quality) — default
#   FULL=1   runs full training (slow, production quality)

set -euo pipefail

# ─────────────────────────────────────────────
# 0.  Configuration
# ─────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/.venv/bin/python"

IMAGE_DIR="${1:-${SCRIPT_DIR}/samples/images/CTC-360-resized/test_gemini}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/benchmark_results_6scenarios}"

FULL="${FULL:-1}"

# 6 log-spaced lambda values between 1e-2 and 1e-4
# Equivalent to: np.logspace(-2, -4, 6)
LAMBDAS=(
    "0.01"
    #"0.003981"
    #"0.001585"
    "0.000631"
    #"0.000251"
    "0.0001"
)

# ── Scenarios ────────────────────────────────────────────────────────────────
# Each entry has the format:  "label|extra_flags|erp_ctx|loss_fn|gauss_weights"
#
#   label         — short identifier written to the TSV
#   flags         — extra arguments forwarded to cc_encode.py
#   erp_ctx       — 0 (rectangular mask) or 1 (geodesic ERP context)
#   loss_fn       — mse or ws_mse
#   gauss_weights — 0 (raw context) or 1 (Gaussian-weighted context)
# ─────────────────────────────────────────────────────────────────────────────
MODES=(
    #"standard||0|mse|0"
    #"erp|--erp_residue|1|mse|0"
    "erp_gw|--erp_residue --erp_gaussian_weights_residue|1|mse|1"
    #"ws_mse|--tune ws_mse|0|ws_mse|0"
    "ws_mse_erp|--erp_residue --tune ws_mse|1|ws_mse|0"
    "ws_mse_erp_gw|--erp_residue --erp_gaussian_weights_residue --tune ws_mse|1|ws_mse|1"
)

# ─────────────────────────────────────────────
# 1.  Preparation
# ─────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

# Aggregate TSV — written once with header, then rows are appended
AGG_TSV="${OUTPUT_DIR}/benchmark_all.tsv"

# Write (or overwrite) the header
# Columns: image_name, mode, erp_ctx, loss_fn, gauss_weights,
#          lmbda, loss, psnr_db, rate_bpp, ws_psnr_db,
#          n_pixels, display_order, coding_order
printf "%-14s\t%-14s\t%-10s\t%-10s\t%-14s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-14s\t%-12s\n" \
    "image_name" "mode" "erp_ctx" "loss_fn" "gauss_weights" "lmbda" \
    "loss" "psnr_db" "rate_bpp" "ws_psnr_db" \
    "n_pixels" "display_order" "coding_order" \
    > "${AGG_TSV}"

if [[ "${FULL}" == "1" ]]; then
    EXTRA_FLAGS=""
    echo "Full training mode (FULL=1). This may take a long time."
else
    EXTRA_FLAGS="--debug"
    echo "Debug mode (FULL=0). Set FULL=1 for production-quality results."
fi

# ─────────────────────────────────────────────
# 2.  Helper: read a field from a results TSV
# ─────────────────────────────────────────────

read_field() {
    local tsv="$1"
    local field="$2"
    awk -v f="${field}" '
        BEGIN { col=0 }
        NR==1 { for(i=1;i<=NF;i++) if($i==f) {col=i;} }
        NR==2 { if(col>0) print $col; else print "" }
    ' "${tsv}"
}

# ─────────────────────────────────────────────
# 3.  Collect images
# ─────────────────────────────────────────────

mapfile -t IMAGES < <(find "${IMAGE_DIR}" -maxdepth 1 \( -name "*.png" -o -name "*.yuv" \) | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: No .png or .yuv images found in ${IMAGE_DIR}"
    exit 1
fi

echo ""
echo "Found ${#IMAGES[@]} image(s):"
for img in "${IMAGES[@]}"; do echo "  ${img}"; done
echo ""
echo "Lambda values: ${LAMBDAS[*]}"
echo "Scenarios    : standard | erp | erp_gw | ws_mse | ws_mse_erp | ws_mse_erp_gw"
echo "Output TSV   : ${AGG_TSV}"
echo "══════════════════════════════════════════════════════════════════"

# ─────────────────────────────────────────────
# 4.  Main loop
# ─────────────────────────────────────────────

TOTAL=$(( ${#IMAGES[@]} * ${#MODES[@]} * ${#LAMBDAS[@]} ))
RUN=0

for IMAGE in "${IMAGES[@]}"; do
    IMAGE_BASENAME="$(basename "${IMAGE%.*}")"   # strip path and extension

    for MODE_ENTRY in "${MODES[@]}"; do
        # Parse the five pipe-separated fields: label|flags|erp_ctx|loss_fn|gauss_weights
        IFS='|' read -r MODE_LABEL MODE_FLAGS MODE_ERP_CTX MODE_LOSS_FN MODE_GAUSS_W <<< "${MODE_ENTRY}"

        for LMBDA in "${LAMBDAS[@]}"; do
            RUN=$(( RUN + 1 ))

            LMBDA_TAG="${LMBDA}"

            WORKDIR="${OUTPUT_DIR}/runs/${IMAGE_BASENAME}/${MODE_LABEL}/lmbda_${LMBDA_TAG}"
            mkdir -p "${WORKDIR}"

            echo ""
            echo "──────────────────────────────────────────────────────────────────"
            echo "  Run ${RUN}/${TOTAL}"
            echo "  Image    : ${IMAGE_BASENAME}"
            echo "  Scenario : ${MODE_LABEL}   [flags: ${MODE_FLAGS:-<none>}]"
            echo "  Lambda   : ${LMBDA}"
            echo "  Workdir  : ${WORKDIR}"
            echo "──────────────────────────────────────────────────────────────────"

            # Run encoder (MODE_FLAGS may contain multiple words — intentionally unquoted)
            # shellcheck disable=SC2086
            "${PYTHON}" "${SCRIPT_DIR}/cc_encode.py" \
                --input   "${IMAGE}" \
                --output  "${WORKDIR}/bitstream.cool" \
                --workdir "${WORKDIR}" \
                --lmbda   "${LMBDA}" \
                ${MODE_FLAGS} \
                ${EXTRA_FLAGS}

            # Locate the generated results TSV (pattern: 0000-results_decoder.tsv)
            RESULT_TSV=$(find "${WORKDIR}" -name "*results_decoder.tsv" | head -1)

            if [[ -z "${RESULT_TSV}" ]]; then
                echo "WARNING: results_decoder.tsv not found in ${WORKDIR}. Skipping."
                continue
            fi

            # Extract metrics
            LOSS=$(read_field     "${RESULT_TSV}" "loss")
            PSNR=$(read_field     "${RESULT_TSV}" "psnr_db")
            RATE=$(read_field     "${RESULT_TSV}" "rate_bpp")
            WS_PSNR=$(read_field  "${RESULT_TSV}" "ws_psnr_db")
            N_PIX=$(read_field    "${RESULT_TSV}" "n_pixels")
            DISP=$(read_field     "${RESULT_TSV}" "display_order")
            COD=$(read_field      "${RESULT_TSV}" "coding_order")

            # Append a tab-separated row to the aggregate TSV
            printf "%-14s\t%-14s\t%-10s\t%-10s\t%-14s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-14s\t%-12s\n" \
                "${IMAGE_BASENAME}" \
                "${MODE_LABEL}" \
                "${MODE_ERP_CTX}" \
                "${MODE_LOSS_FN}" \
                "${MODE_GAUSS_W}" \
                "${LMBDA}" \
                "${LOSS}" \
                "${PSNR}" \
                "${RATE}" \
                "${WS_PSNR}" \
                "${N_PIX}" \
                "${DISP}" \
                "${COD}" \
                >> "${AGG_TSV}"

            echo "  ✓ Appended to ${AGG_TSV}"
        done   # LMBDA
    done       # MODE
done           # IMAGE

# ─────────────────────────────────────────────
# 5.  Summary
# ─────────────────────────────────────────────

echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  BENCHMARK COMPLETE"
echo "══════════════════════════════════════════════════════════════════"
echo ""
echo "All ${RUN} runs finished. Aggregate results:"
echo "  ${AGG_TSV}"
echo ""
column -t "${AGG_TSV}"

