# Rotation-Equivariant Conditional Spherical Neural Fields for Learning a Natural Illumination Prior
![teaser](data/other/teaser.gif)
## [Project page](https://jadgardner.github.io/RENI.html) |  [Paper](https://arxiv.org/abs/2206.03858) | [Data](https://drive.google.com/drive/folders/1pMx2oolATFSRIZ2iRc9x2cNrqZQDB1En?usp=sharing)
This is the official repo for the implementation of **RENI: A Rotation-Equivariant Natural Illumination Model**.

If you use our code, please cite the following paper:

```
@inproceedings{
  gardner2022rotationequivariant,
  title={Rotation-Equivariant Conditional Spherical Neural Fields for Learning a Natural Illumination Prior},
  author={James A D Gardner and Bernhard Egger and William A P Smith},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=cj6K4IWVomU}
}
```

## News:
**01/11/2022**: Updated code! Now implemented using PyTorch-Lightning. Refactored code makes training and using RENI in downstream tasks easier.

**15/09/2022**: Accepted to NeurIPS 2022!!! 
## Setup:
1\. Clone this repository:
```shell
git clone https://github.com/JADGardner/RENI.git
```
2\. Setup conda environment:
```shell
cd RENI
conda env create -f environment.yml
conda activate reni
```
4\. You can download the RENI dataset and pre-trained models using setup.py
```shell
python setup.py
```
5\. You can train a RENI from scratch by setting the hyperparameters in [configs/experiment.yaml](configs/experiment.yaml) and running the [run.py](run.py) script:
```shell
python run.py --cfg_path configs/experiment.yaml --gpus '0, 1, 2, 3'
```
6\. The [example notebook](examples.ipynb) demonstrates using a pre-trained RENI in a downstream task as a prior for environment map in-painting.
