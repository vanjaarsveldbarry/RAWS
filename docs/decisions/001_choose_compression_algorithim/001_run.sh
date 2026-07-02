#!/bin/bash
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=72:00:00
#SBATCH --exclusive
#SBATCH -J compression_benchmark
#SBATCH -o /eejit/home/7006713/projects/RAWS/RAWS/docs/decisions/001_choose_compression_algorithim.out


cd "/eejit/home/7006713/projects/RAWS/RAWS/docs/decisions/001_choose_compression_algorithim"
script_dir=$(realpath .)
root_dir=$(realpath "$script_dir/../../../")
output_dir="/scratch/depfg/7006713/RAWS/001_choose_compression_algorithim"
mkdir -p $output_dir/input $output_dir/output $output_dir/_temp

eval "$(pixi shell-hook --manifest-path "$root_dir")"


download_discharge_year() {
    local BASE_URL="https://geo.public.data.uu.nl/vault-geowat-simulations/esd_1km_PCRGLOBWB%5B1730451774%5D/original/monthly"
    local YEAR=1985
    local OUT_DIR=$output_dir/input/discharge

    mkdir -p "${OUT_DIR}"
    echo "Downloading ${YEAR} → ${OUT_DIR}"

    local SUCCESS=0
    local FAILED=0

    for MM in $(seq -w 1 12); do
        local FILENAME="discharge_monthAvg_${YEAR}${MM}_output.nc"
        local URL="${BASE_URL}/${FILENAME}"
        local OUT_FILE="${OUT_DIR}/${FILENAME}"

        echo "  [get]  ${FILENAME}..."
        if wget --show-progress -O "${OUT_FILE}" "${URL}" 2>&1; then
            (( SUCCESS++ ))
        else
            (( FAILED++ ))
        fi
    done

    echo "  [done] ${SUCCESS} succeeded, ${FAILED} failed"
}

concat_discharge() {
    local IN_DIR="$output_dir/input/discharge"
    local OUT_FILE="$output_dir/input/discharge_1985_uncompressed.nc"

    echo "Concatenating discharge files → ${OUT_FILE}"
    cdo -f nc4 cat "${IN_DIR}"/discharge_monthAvg_*.nc "${OUT_FILE}"
    echo "  [done] $(du -sh "${OUT_FILE}" | cut -f1)"
}

# download_discharge_year

# concat_discharge

# taskset -c 0-95 python "$script_dir/compression_benchmark.py" \
#     --input-dir "$output_dir/input" \
#     --output-dir "$output_dir/output" \
#     --temp-dir "$output_dir/_temp" \
#     --n-workers 24 \
#     --worker-memory-limit 7GB

# taskset -c 0-95 python "$script_dir/compare_benchmarks.py" \
#     --input-dir "$output_dir/input" \
#     --output-dir "$output_dir/output" \
#     --temp-dir "$output_dir/_temp"

taskset -c 0-95 python "$script_dir/plot_results.py" \
    --input-dir "$output_dir/input" \
    --output-dir "$output_dir/output" \
    --temp-dir "$output_dir/_temp"