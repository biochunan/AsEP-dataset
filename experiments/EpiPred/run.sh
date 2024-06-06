#!/bin/zsh
# aims to run EpiPred on one AbAg complex


# mutable args
abdbId="1a14_0P"
agChain="N"

# immutable args
BASE=$(dirname $(realpath $0))
num_epis=3

docker run --rm \
    -v ${BASE}/epipred-input:/EpiPred/epipred-input \
    -v ${BASE}/epipred-output:/EpiPred/epipred-output \
    -w /EpiPred \
    ${USER}/epipred-root:runtime-v1.1 \
    --file_ab ./epipred-input/pdb${abdbId}.pdb \
    --file_ag ./epipred-input/pdb${abdbId}.pdb \
    --chains_ab HL --chains_ag ${agChain} \
    --job_id epipred-output/${abdbId}_${agChain} --num_epis ${num_epis}
