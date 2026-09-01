#!/usr/bin/env bash
# run_benchmark.sh
#
# Runs all combinations of:
#   - Images  : every *.png / *.yuv found in IMAGE_DIR
#   - Modes   : standard (no --erp_residue) and erp (--erp_residue)
#   - Lambdas : 6 log-spaced values between 1e-2 and 1e-4
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

IMAGE_DIR="${1:-${SCRIPT_DIR}/samples/images/CTC-360-resized}"
OUTPUT_DIR="${2:-${SCRIPT_DIR}/benchmark_results_ctc_poles}"

FULL="${FULL:-1}"

# 6 log-spaced lambda values between 1e-2 and 1e-4
# Equivalent to: np.logspace(-2, -4, 6)
LAMBDAS=(
    "0.01"
    "0.003981"
    "0.001585"
    "0.000631"
    "0.000251"
    "0.0001"
)

# Modes: each entry is "label|extra_flags"
MODES=(
    #"standard|"
    #"erp|--erp_residue"
    "erp_polar30|--erp_residue --erp_polar_threshold_deg_residue 30.0"
)

# ─────────────────────────────────────────────
# 1.  Preparation
# ─────────────────────────────────────────────

mkdir -p "${OUTPUT_DIR}"

# Aggregate TSV — written once with header, then rows are appended
AGG_TSV="${OUTPUT_DIR}/benchmark_all.tsv"

# Write (or overwrite) the header
printf "%-14s\t%-10s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-14s\t%-12s\n" \
    "image_name" "mode" "lmbda" \
    "loss" "psnr_db" "rate_bpp" \
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
        NR==1 { for(i=1;i<=NF;i++) if($i==f) col=i }
        NR==2 { print $col }
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
echo "Modes        : standard, erp"
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
        MODE_LABEL="${MODE_ENTRY%%|*}"            # "standard" or "erp"
        MODE_FLAGS="${MODE_ENTRY##*|}"            # "" or "--erp_residue"

        for LMBDA in "${LAMBDAS[@]}"; do
            RUN=$(( RUN + 1 ))

            LMBDA_TAG="${LMBDA}"

            WORKDIR="${OUTPUT_DIR}/runs/${IMAGE_BASENAME}/${MODE_LABEL}/lmbda_${LMBDA_TAG}"
            mkdir -p "${WORKDIR}"

            echo ""
            echo "──────────────────────────────────────────────────────────────────"
            echo "  Run ${RUN}/${TOTAL}"
            echo "  Image  : ${IMAGE_BASENAME}"
            echo "  Mode   : ${MODE_LABEL}"
            echo "  Lambda : ${LMBDA}"
            echo "  Workdir: ${WORKDIR}"
            echo "──────────────────────────────────────────────────────────────────"

            # Run encoder (MODE_FLAGS may be empty — no quotes to avoid passing empty arg)
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
            N_PIX=$(read_field    "${RESULT_TSV}" "n_pixels")
            DISP=$(read_field     "${RESULT_TSV}" "display_order")
            COD=$(read_field      "${RESULT_TSV}" "coding_order")

            # Append a tab-separated row to the aggregate TSV
            printf "%-14s\t%-10s\t%-12s\t%-12s\t%-12s\t%-12s\t%-12s\t%-14s\t%-12s\n" \
                "${IMAGE_BASENAME}" \
                "${MODE_LABEL}" \
                "${LMBDA}" \
                "${LOSS}" \
                "${PSNR}" \
                "${RATE}" \
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
