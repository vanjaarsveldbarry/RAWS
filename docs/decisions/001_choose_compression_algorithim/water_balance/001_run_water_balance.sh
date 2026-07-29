#!/bin/bash
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00
#SBATCH -J dw_water_balance
#SBATCH -o /projects/prjs1222/RAWS/RAWS/docs/decisions/001_choose_compression_algorithim/water_balance/001_choose_compression_algorithim.out


cd "/projects/prjs1222/RAWS/RAWS/docs/decisions/001_choose_compression_algorithim/water_balance"
script_dir=$(realpath .)
root_dir=$(realpath "$script_dir/../../../../")
output_dir="/projects/prjs1222/RAWS/RAWS/output/001_choose_compression_algorithim/water_balance"
mkdir -p $output_dir/input $output_dir/output $output_dir/_temp

eval "$(pixi shell-hook --manifest-path "$root_dir")"

download_hydrobasins() {
    local target_dir="${1:?usage: download_hydrobasins <target_dir>}"
    local docs_dir="$target_dir/docs"
    local standard_dir="$target_dir/standard"

    command -v wget >/dev/null 2>&1 || { echo "Error: wget is required but not installed." >&2; return 1; }
    command -v unzip >/dev/null 2>&1 || { echo "Error: unzip is required but not installed." >&2; return 1; }

    mkdir -p "$docs_dir" "$standard_dir"

    wget -c --show-progress --no-verbose -P "$docs_dir" \
        "https://data.hydrosheds.org/file/technical-documentation/HydroBASINS_TechDoc_v1c.pdf"

    local continents=(af ar as au eu gr na sa si)
    local base_url="https://data.hydrosheds.org/file/hydrobasins/standard"
    local continent
    for continent in "${continents[@]}"; do
        wget -c --show-progress --no-verbose -P "$standard_dir" \
            "${base_url}/hybas_${continent}_lev01-12_v1c.zip"
    done

    # Extract each continent zip into its own subfolder under standard/
    for continent in "${continents[@]}"; do
        local zip_file="$standard_dir/hybas_${continent}_lev01-12_v1c.zip"
        local extract_dir="$standard_dir/${continent}"
        mkdir -p "$extract_dir"
        unzip -o -q "$zip_file" -d "$extract_dir"
    done

    echo "HydroBASINS download summary:"
    echo "  docs/:     $(find "$docs_dir" -type f | wc -l) file(s), $(du -sh "$docs_dir" | cut -f1)"
    echo "  standard/: $(find "$standard_dir" -type f | wc -l) file(s), $(du -sh "$standard_dir" | cut -f1)"
}
# cp /projects/prjs1222/globgm_input/_data/globgm_input/cdo_grid_area_30sec_map_correct_lat.nc "$output_dir/input/cdo_grid_area_30sec_map_correct_lat.nc"

# download_hydrobasins "$output_dir/input/hydrobasins"

# python "$script_dir/process_water_provinces.py"
# python "$script_dir/dw_data.py" "$output_dir/_temp"
# python "$script_dir/merge_forcing.py" "$output_dir"

taskset -c 0-191 python "$script_dir/compression_benchmark.py" \
    --output-dir "$output_dir" \
    --n-workers 48 \
    --worker-memory-limit 7GB

taskset -c 0-191 python "$script_dir/compare_benchmarks.py" \
    --output-dir "$output_dir" \
    --levels "3,6,8,12" \
    --closure-tol-pp 0.1

taskset -c 0-191 python $script_dir/plot_results.py \
    --output-dir "$output_dir"

