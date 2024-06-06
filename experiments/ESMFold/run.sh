#!/bin/zsh

docker run --gpus all --rm \
    -v ./input:/input \
    -v ./output:/output \
    biochunan/esmfold-image:latest -i /input/walle1723.fasta -o /output/ > ./logs/pred.log 2>./logs/pred.err
    # esmfold:base -i /input/seq.fasta -o /output/ > ./logs/pred.log 2>&1
    # esmfold:base -i /input/seq.fasta -o /output/ --chunk-size 128 --cpu-offload > ./logs/pred.log 2>&1
