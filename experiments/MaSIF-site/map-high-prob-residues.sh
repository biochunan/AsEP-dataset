#!/bin/zsh

# Aim: map high probability vertices to residues in the input PDB file

# immutable variables
WD=$(dirname $(realpath $0))
data_preparation="${WD}/masif_output/data_preparation"
output="${WD}/masif_output/pred_output"
OUT="${WD}/mapped_residues"

# muatble variables
job_name="1a14_N"
prob_thr=${1:-0.7}
radius=${2:-1.2}

parallel -j 16 --bar --joblog ./parallel_run.log --colsep ',' \
    "python map-highprob-vertices-to-residues.py \
    --job_name {} \
    --data_preparation  ${data_preparation} \
    --output ${output} \
    --prob_thr ${prob_thr} \
    --radius ${radius} > mapped_residues/{}.txt" \
    ::: $(cat jobids.txt)
