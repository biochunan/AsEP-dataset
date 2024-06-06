#!/bin/zsh

mkdir -p masif_output/data_preparation
mkdir -p masif_output/pred_output
mkdir -p masif_output/logs

parallel -j 24 --bar --joblog ./masif_output/parallel_run.log --colsep ',' \
  docker run --rm \
  -v /home/chunan/Dataset/AbDb/abdb_newdata_20220926:/dataset/abdb \
  -v ./masif_output/data_preparation:/masif/data/masif_site/data_preparation  \
  -v ./masif_output/pred_output:/masif/data/masif_site/output  \
  -v ./masif_output/logs:/masif/data/masif_site/logs  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  './data_prepare_one.sh  --file /dataset/abdb/pdb{1}.mar {2} > ./masif_output/logs/data_preparation/{2}.log 2>./masif_output/logs/data_preparation/{2}.err' ::: `cat ./evaluate_epitope_pred/ag_chain_ids.txt  | awk -F ',' '{print $1","$2}' `

# predict_site
parallel -j 24 --bar --joblog ./masif_output/parallel_run.log --colsep ',' \
  docker run --rm \
  -v /home/chunan/Dataset/AbDb/abdb_newdata_20220926:/dataset/abdb \
  -v ./masif_output/data_preparation:/masif/data/masif_site/data_preparation  \
  -v ./masif_output/pred_output:/masif/data/masif_site/output  \
  -v ./masif_output/logs:/masif/data/masif_site/logs  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  './predict_site.sh {2} >./masif_output/logs/predict_site/{2}.log 2>./masif_output/logs/predict_site/{2}.err' ::: `cat ./evaluate_epitope_pred/ag_chain_ids.txt  | awk -F ',' '{print $1","$2}' `


# color_site
parallel -j 24 --bar --joblog ./masif_output/parallel_run.log --colsep ',' \
  docker run --rm \
  -v /home/chunan/Dataset/AbDb/abdb_newdata_20220926:/dataset/abdb \
  -v ./masif_output/data_preparation:/masif/data/masif_site/data_preparation  \
  -v ./masif_output/pred_output:/masif/data/masif_site/output  \
  -v ./masif_output/logs:/masif/data/masif_site/logs  \
  -w /masif/data/masif_site \
  pablogainza/masif \
  './color_site.sh {2} > ./masif_output/logs/color_site/{2}.log 2>./masif_output/logs/color_site/{2}.err' ::: `cat ./evaluate_epitope_pred/ag_chain_ids.txt  | awk -F ',' '{print $1","$2}' `