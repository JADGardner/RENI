# Rotation-Equivariant Conditional Spherical Neural Fields for Learning a Natural Illumination Prior
![teaser](imgs/teaser.gif)
## [Project page](https://jadgardner.github.io/RENI.html) |  [Paper](https://arxiv.org/abs/2206.03858) | [Data](https://drive.google.com/drive/folders/1pMx2oolATFSRIZ2iRc9x2cNrqZQDB1En?usp=sharing)
This is the official repo for the implementation of **RENI: A Rotation-Equivariant Natural Illumination Model**.

## Setup:
1\. Clone this repository:
```shell
git clone https://github.com/JADGardner/RENI.git
```
2\. Setup environment and permissions for launch script:
```shell
cd RENI
conda env create -f environment.yml
conda activate reni
chmod +x ./launch_scripts/run_training.sh 
```
3\. Set up [wandb](https://docs.wandb.ai/quickstart) for experiment tracking: 
* a. [Sign up](https://wandb.ai/site) for a free account at https://wandb.ai/site and then login to your wandb account.
* b. Login to the wandb library on your machine. You will find your API key here: https://wandb.ai/authorize
```shell
wandb login
```
4\. Download the datasets from [here](https://drive.google.com/drive/folders/1pMx2oolATFSRIZ2iRc9x2cNrqZQDB1En?usp=sharing) and place the unzipped HDRI_Test and HDRI_Train folders in `data/processed/`

5\. You can quickly produce a set of results by running:
```shell
cd src
python demo.py
```
<br/>

## Running:
### Fitting a model to unseen images:
1\. Setup `training_configs/config_main.yaml`:
```yaml
previous_run_path: PATH_TO_PREVIOUS_RUN
test_dataset_path: PATH_TO_TEST_DATASET
latent_code_optimisation: TRUE/FALSE
inverse_rendering_task: TRUE/FALSE
```
* e.g.
```yaml
previous_run_path: models/RENIVariationalAutoDecoder/network_128_5/ndims_36/
test_dataset_path: data/processed/HDRI_Test/
latent_code_optimisation: True
inverse_rendering_task: False
```
2\. Setup `training_configs/config_latent.yaml`:
```yaml
latent_batch_size: INT
apply_mask: TRUE/FALSE
mask_path: PATH_TO_MASK
```
* e.g.
```yaml
latent_batch_size: 21
apply_mask: False
mask_path: data/processed/Masks/Mask-3.png
```
3\. Launch the run:
```shell
./launch_scripts/run_training.sh
```
To fit to your own HDR equirectangular images, ensure they are in the .exr format and update `test_dataset_path` to the folder containing your images.

<br/>

### Fitting a model in the inverse rendering task:
1\. Setup `training_configs/config_main.yaml`:
```yaml
previous_run_path: PATH_TO_PREVIOUS_RUN
test_dataset_path: PATH_TO_TEST_DATASET
latent_code_optimisation: TRUE/FALSE
inverse_rendering_task: TRUE/FALSE
```
* e.g.
```yaml
previous_run_path: models/RENIVariationalAutoDecoder/network_128_5/ndims_36/
test_dataset_path: data/processed/HDRI_Test/
latent_code_optimisation: False
inverse_rendering_task: True
```
2\. Setup `training_configs/config_inverse.yaml`:
```yaml
object_path: PATH_TO_OBJECT
kd_value: FLOAT
```
* e.g.
```yaml
object_path: data/graphics/3d_models/teapot.obj
kd_value: 1.0
```
3\. Launch the run:
```shell
./launch_scripts/run_training.sh
```
<br/>

### Training a new model:
1\. Setup `training_configs/config_main.yaml`:
```yaml
previous_run_path: None
dataset_path: PATH_TO_TRAINING_DATASET
test_dataset_path: PATH_TO_TEST_DATASET
latent_code_optimisation: TRUE/FALSE
inverse_rendering_task: TRUE/FALSE
model_type: RENIVariationalAutoDecoder/RENIAutoDecoder
ndims: INT
RENI_hidden_layers: INT
RENI_hidden_features: INT
last_layer_linear: TRUE/FALSE
```
* e.g.
```yaml
previous_run_path: None
dataset_path: data/processed/HDRI_Train/
test_dataset_path: data/processed/HDRI_Test/
latent_code_optimisation: True
inverse_rendering_task: False
model_type: RENIVariationalAutoDecoder
ndims: 36
RENI_hidden_layers: 5
RENI_hidden_features: 128
last_layer_linear: True
```
2\. Launch the run:
```shell
./launch_scripts/run_training.sh
```