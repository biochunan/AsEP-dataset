# Benchmark ESMFold

## Docker image

Refer to this repo for ESMFold docker image: [esmfold-docker-image](https://github.com/biochunan/esmfold-docker-image)

You can pull this image from Docker Hub:

```shell
docker pull biochunan/esmfold-image
```

## Run ESMFold from host machine

We provide a shell script `run.sh` to run ESMFold for the entire `AsEP` dataset.

## Evaluation

Use the provided script `evaluate.py` to evaluate the prediction results.

NOTE: remember to use the correct path to the ABDB dataset. See the dependencies section for more details.

## Dependencies

- `clustal-omega`

Install `clustal-omega` via conda:

```shell
conda install -c bioconda clustalo
```

- `ABDB` This is the source dataset we used to create the `AsEP` dataset. You can download it from [ABDB](http://www.abybank.org/abdb/), the snapshot used in the paper is available at: (1) [AbDb-20220926.tar.gz](https://drive.google.com/file/d/1kAgSOjYBqb02IIEsc9yhJNiaoWRhpoaL/view?usp=drive_link) or (2) [abdb_20220926.zip](http://www.abybank.org/abdb/snapshots/abdb_20220926.zip)
- change the variable ABDB in `evaluate.py` to the path of the downloaded ABDB dataset.