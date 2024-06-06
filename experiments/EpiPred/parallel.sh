#!/bin/zsh

# Aim: run EpiPred on one AbAg complex in AsEP

BASE=$(dirname $(realpath $0))
INPUT_PATH=${BASE}/epipred-input
TASKLIST=${IBASE}/assets/ag-chain-ids.txt
num_epis=3

parallel -j 20 --bar --joblog ./parallel_run.log --colsep ',' \
docker run --rm \
    -v ${INPUT_PATH}:/EpiPred/epitope-input \
    -v ${WD}/epipred-output:/EpiPred/epipred-output \
    -v ${WD}/logs:/EpiPred/logs \
    -w /EpiPred \
    epipred:latest \
    'python EpiPred.py --file_ab ./epitope-input/pdbs/pdb{1}.pdb \
    --file_ag ./epitope-input/pdbs/pdb{1}.pdb \
    --chains_ab HL \
    --chains_ag {3} \
    --job_id epipred-output/{2} \
    --num_epis 5 > logs/{2}.log 2>&1' \
    ::: `cat ${TASKLIST}  | awk -F ',' '{print $1","$2","$3}' `
