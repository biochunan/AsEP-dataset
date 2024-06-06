# MaSIF-Site

contents:

- [MaSIF-Site](#masif-site)
  - [Dataset](#dataset)
  - [MaSIF-site Container Documentation](#masif-site-container-documentation)
  - [Run MaSIF-site pipeline from host machine](#run-masif-site-pipeline-from-host-machine)
  - [Run MaSIF-site pipeline inside the container](#run-masif-site-pipeline-inside-the-container)
    - [Step 1 - prepare data for inference](#step-1---prepare-data-for-inference)
    - [Step 2 - Running inference to predict interaction sites](#step-2---running-inference-to-predict-interaction-sites)
    - [Step 3 - Visualize the predicted sites](#step-3---visualize-the-predicted-sites)
  - [File description](#file-description)
  - [Evaluation](#evaluation)
    - [Mapping high probability vertices to residues](#mapping-high-probability-vertices-to-residues)
    - [Evaluate](#evaluate)

Benchmark MaSIF-site on the WALLE1.0 dataset.

MaSIF-site takes a single antigen structure as input, and outputs the surface patches that are most likely to be involved in the binding of the antigen to its partner.

## Dataset

For WALLE1.0, need to extract antigen chains from the PDB files.

## MaSIF-site Container Documentation

Use the docker container is easier. Its documentation is accessible at [docker_tutorial.md](https://github.com/LPDI-EPFL/masif/blob/master/docker_tutorial.md)

## Run MaSIF-site pipeline from host machine

The docker image is downloadable from :link: [masif-site.tar](https://drive.google.com/open?id=1gCgYwPPyzoAMRV_DG69Edx80CZs5ij3V&usp=drive_fs)

We provide a shell script `run.sh` to run the MaSIF-site pipeline, for example:

```shell
zsh run-dev.sh \
-i 1a14_0P \
-n 1a14_N \
-o ./example/masif_output \
-d /mnt/bob/shared/AbDb/abdb_newdata_20220926
```

- `-i 1a14_0P`: input AbDb file ID, this is uesd to locate the AbDb file
- `-n 1a14_N`: format `<pdb_code>_<antigen_chainID>` you can replace the `<pdb_code>` with a custom name, but make sure the antigen chain is correct.

Antigen chains of each item is provided in `jobids.txt` file with the format `<pdb_code>_<antigen_chainID>`for example:

```plaintext
1a14_N
1adq_A
1afv_B
1ahw_F
...
```

We also provide a parallelized version of the script `parallel.sh` to run the pipeline in parallel.

```shell
zsh parallel.sh
```

## Run MaSIF-site pipeline inside the container

Run it as a container with an interface

```sh
docker run -it --name masif-site \
  -v /home/chunan/Dataset/AbDb/abdb_newdata_20220926:/dataset/abdb \
  -v ./masif_output:/masif/data/masif_site/data_preparation  \
  -w /masif/data/masif_site
```

- `-it`: interactive mode
- `--name`: name of the container e.g. `masif-site` here, making it easier to start/stop the container
- `-v`: mount the local directory to the container directory
  - here we specified two mount points, one for the dataset, one for the output
- `-w`: set the working directory
  - this is the directory where the container will start

### Step 1 - prepare data for inference

Inside the container, run the following command to prepare the data

```sh
# root@bb75bbc88f52:/masif/data/masif_site
$ ./data_prepare_one.sh --file /dataset/abdb/pdb1cz8_1P.mar 1cz8_HL
```

This creates a number of files under folder `data_preparation/1cz8_HL`

- `00-reaw_pdbs/1cz8.pdb`: the raw pdb file of the antigen
- `01-benchmark_pdbs/1cz8_HL.pdb`: the pdb file with the specified chains, in this case, H and L chains
- `01-benchmark_surfaces/1cz8_HL.ply`: the surface file
- `04a-precomputation_9A/precomputation/1cz8_HL/`: the precomputed data for the surface file

stdout

```plaintext
Running masif site on /dataset/abdb/pdb1cz8_1P.mar
Empty
Removing degenerated triangles
Removing degenerated triangles
1cz8_HL
Reading data from input ply surface files.
Dijkstra took 2.50s
Only MDS time: 9.69s
Full loop time: 16.06s
MDS took 16.06s

```

### Step 2 - Running inference to predict interaction sites

```sh
# root@bb75bbc88f52:/masif/data/masif_site
$ ./predict_site.sh 1cz8_HL
```

This creates a file `output/all_feat_3l/pred_data/1cz8_HL.npz` which contains the predicted scores for each patch.

<details>

<summary> stdout (long output) </summary>

```txt
Setting model_dir to nn_models/all_feat_3l/model_data/
Setting feat_mask to [1.0, 1.0, 1.0, 1.0, 1.0]
Setting n_conv_layers to 3
Setting out_pred_dir to output/all_feat_3l/pred_data/
Setting out_surf_dir to output/all_feat_3l/pred_surfaces/
(12, 2)
WARNING:tensorflow:From /masif/source/masif_modules/MaSIF_site.py:108: calling reduce_sum (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
Instructions for updating:
keep_dims is deprecated, use keepdims instead
global_desc shape: <unknown>
2023-11-22 14:55:10.979661: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
<tf.Variable 'mu_rho_0:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_0:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_0:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_0:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_1:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_1:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_1:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_1:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_4:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_4:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_4:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_4:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_l2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_l2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_l2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_l2:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_rho_l3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'mu_theta_l3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_rho_l3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'sigma_theta_l3:0' shape=(1, 12) dtype=float32_ref>
12
<tf.Variable 'b_conv_0:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'b_conv_1:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'b_conv_2:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'b_conv_3:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'b_conv_4:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'W_conv_0:0' shape=(12, 12) dtype=float32_ref>
144
<tf.Variable 'W_conv_1:0' shape=(12, 12) dtype=float32_ref>
144
<tf.Variable 'W_conv_2:0' shape=(12, 12) dtype=float32_ref>
144
<tf.Variable 'W_conv_3:0' shape=(12, 12) dtype=float32_ref>
144
<tf.Variable 'W_conv_4:0' shape=(12, 12) dtype=float32_ref>
144
<tf.Variable 'fully_connected/weights:0' shape=(60, 12) dtype=float32_ref>
720
<tf.Variable 'fully_connected/biases:0' shape=(12,) dtype=float32_ref>
12
<tf.Variable 'fully_connected_1/weights:0' shape=(12, 5) dtype=float32_ref>
60
<tf.Variable 'fully_connected_1/biases:0' shape=(5,) dtype=float32_ref>
5
<tf.Variable 'W_conv_l2:0' shape=(60, 60) dtype=float32_ref>
3600
<tf.Variable 'b_conv_l2:0' shape=(60,) dtype=float32_ref>
60
<tf.Variable 'W_conv_l3:0' shape=(60, 60) dtype=float32_ref>
3600
<tf.Variable 'b_conv_l3:0' shape=(60,) dtype=float32_ref>
60
<tf.Variable 'fully_connected_2/weights:0' shape=(5, 4) dtype=float32_ref>
20
<tf.Variable 'fully_connected_2/biases:0' shape=(4,) dtype=float32_ref>
4
<tf.Variable 'fully_connected_3/weights:0' shape=(4, 2) dtype=float32_ref>
8
<tf.Variable 'fully_connected_3/biases:0' shape=(2,) dtype=float32_ref>
2
Total number parameters: 9267
Restoring model from: nn_models/all_feat_3l/model_data/model
1cz8_HL
Evaluating 1cz8_HL
Total number of patches:4346

Total number of patches for which scores were computed: 4346

GPU time (real time, not actual GPU time): 0.708s
```

</details>

### Step 3 - Visualize the predicted sites

```sh
# root@ef9a7fdacb45:/masif/data/masif_site
$ ./color_site.sh 1cz8_HL
```

stdout

```plaintext
Setting model_dir to nn_models/all_feat_3l/model_data/
Setting feat_mask to [1.0, 1.0, 1.0, 1.0, 1.0]
Setting n_conv_layers to 3
Setting out_pred_dir to output/all_feat_3l/pred_data/
Setting out_surf_dir to output/all_feat_3l/pred_surfaces/
ROC AUC score for protein 1cz8_HL : 0.88
Successfully saved file output/all_feat_3l/pred_surfaces/1cz8_HL.ply
Computed the ROC AUC for 1 proteins
Median ROC AUC score: 0.8829849087377829

```

This creates a file `output/all_feat_3l/pred_surfaces/1cz8_HL.ply` which contains the predicted scores for each patch.

## File description

- `masif_output`: output from MaSIF-site including data preparation and prediction
- `mapped_residues`: mapped MaSIF-site predicted vertices to the residues in the original pdb file with high probability
- `evaluate_epitope_pred`: prepare the input to MaSIF-site, including abdbids and job_ids (essentially just pdb_code + antigen chain ID)
- `assets/jobids.txt`: list of job IDs that MaSIF-site predicted, the 2nd column in the file `ag-chain-ids.txt` under `evaluate_epitope_pred`, e.g. `6wzl_F`
- `assets/failed.txt`: list of abdbids that MaSIF-site failed to predict


## Evaluation

### Mapping high probability vertices to residues

The mapping is done by the following steps:

1. Extract the vertices with high probability (thr: 0.7) from the `pred_<job_name>.npy` file
2. Find the residues located within thr (default: 1.2Å) of the vertices
3. Output the mapping to a file

We provide a script `map-high-prob-residues.sh` to do this mapping.

### Evaluate

To evaluate the performance of MaSIF-site, use the provided python script `evaluate.py`, this will output a json file containing all metrics for each sample in the dataset.
