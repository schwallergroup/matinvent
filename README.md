# MatInvent: Accelerating inverse materials design using generative diffusion models with reinforcement learning

[![arXiv](https://img.shields.io/badge/arXiv-2511.03112-b31b1b.svg)](https://arxiv.org/abs/2511.03112)
<!-- [![DOI](https://img.shields.io/badge/DOI-10.1038/s41467--025--58499--7-blue.svg)](https://www.nature.com/articles/s41467-025-58499-7) -->

MatInvent is a general and efficient reinforcement learning workflow that optimizes diffusion models for goal-directed crystal generation. MatInvent enables robust optimization for inverse material design tasks with single or multiple target properties. Compatible with diverse diffusion model architectures and property constraints, MatInvent could offer broad applicability in materials discovery.


## 🚀 Environment Setup
- System requirements: This package requires a standard Linux computer with GPU (supports CUDA >= 11) and enough RAM (> 2 GB). If you want to run the code on a GPU that does not support CUDA>=11, you need to modify the versions of PyTorch and CUDA in the [env.yml](env.yml) file.
- We'll use `conda` to install dependencies and set up the environment for a Nvidia GPU machine.
We recommend using the [Miniconda installer](https://www.anaconda.com/docs/getting-started/miniconda/install). You can also install [`mamba`](https://mamba.readthedocs.io/en/latest/) to the conda base environment by the command `conda install mamba -n base -c conda-forge`. `mamba` is a faster, drop-in replacement for `conda`.
- Then create a conda environment and install the dependencies:
    ```bash
    conda env create -f env.yml
    # OR use mamba to create new env faster
    # mamba env create -f env.yml
    ```
    Activate the conda environment with `conda activate matinvent`.
- We use[ Weights & Biases (wandb)](https://wandb.ai/) by default to record RL results, which can visualize RL curves online in real time. If you would like to use `wandb`, please [login](https://wandb.ai/quickstart?product=models) to your account before running the RL experiment. If you do not want to use `wandb`, please change the `logger` parameter in the [run script](scripts/run_rl.sh) to `logger=csv`.
- If you want to calculate specific heat capacities for RL rewards using [FairChem](https://github.com/facebookresearch/fairchem) software and pre-trained ML potentials [eSEN-30M-OAM](https://huggingface.co/facebook/OMAT24), please follow this [tutorial](rewards/calculators/fairchem/README.md) to install the additional conda environment.


## 🤖 Checkpoints
Checkpoint files for the pretrained diffusion models and property prediction model are available at [Hugging Face](https://huggingface.co/jwchen25/MatInvent).


## 🏆 RL rewards and property evaluation
The material properties and rewards can be obtained through theoretical simulations, ML predictions, and empirical calculations. Any cost-acceptable property estimator can be used to calculate RL rewards without requiring gradients. This codebase provides over 15 property evaluators and corresponding RL rewards, encompassing electronic, magnetic, mechanical, thermal, dielectric, and physicochemical properties, as well as synthesizability and supply chain risk. Since DFT computation is expensive and time-consuming, we provide additional ML prediction models for rapid testing. All configuration files related to RL rewards can be found in [`configs/reward`](configs/reward).

## 🔥 RL experiments
You can use the following commands or [run_rl.sh](scripts/run_rl.sh) script to run RL experiments for single or multiple property optimization tasks.
```bash
# option 1
python -u main.py \
    expname=test \
    pipeline=mat_invent \
    model=mattergen \
    reward=hhi \
    logger=wandb
```
```bash
# option 2
bash scripts/run_rl.sh
```
You need to modify the relevant parameters for your own task. The parameter `expname` defines the name of this RL experiment. `model` defines which [diffusion model](configs/model) is used in RL. `reward` defines the [target property](configs/reward) for the RL reward. We use the [hydra](https://github.com/facebookresearch/hydra) package to manage the configuration of input parameters.

[gen_eval.sh](scripts/gen_eval.sh) script is used to generate a large number of crystal structures using unconditional or RL-finetuned [MatterGen](https://github.com/microsoft/mattergen) models and to evaluate the generation quality (such as SUN ratio).
```bash
bash scripts/gen_eval.sh
```
Please modify the environment variables in [scripts/gen_eval.sh](scripts/gen_eval.sh) before running it.


## 🌈 Acknowledgements
This work was supported as part of NCCR Catalysis (grant number 225147), a National Centre of Competence in Research funded by the Swiss National Science Foundation.


## 📝 Citation
If you find our work useful, please consider citing it:
```bibtex
@article{matinvent,
  title={Accelerating inverse materials design using generative diffusion models with reinforcement learning},
  author={Chen, Junwu and Guo, Jeff and Fako, Edvin and Schwaller, Philippe},
  journal={arXiv preprint arXiv:2511.03112},
  year={2025}
}
```