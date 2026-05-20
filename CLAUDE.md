# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MatInvent is a reinforcement learning workflow that fine-tunes diffusion models for goal-directed crystal structure generation. It supports MatterGen and DiffCSP as backbone diffusion models, and optimizes for 15+ material properties (electronic, magnetic, mechanical, thermal, synthesizability, etc.).

Paper: [arXiv:2511.03112](https://arxiv.org/abs/2511.03112) | Checkpoints: [Hugging Face jwchen25/MatInvent](https://huggingface.co/jwchen25/MatInvent)

## Python Environment

Always use the uv virtual environment at `.venv` for all code execution and testing:

```bash
source .venv/bin/activate
# or invoke directly:
.venv/bin/python ...
```

This environment is pre-installed with all dependencies. MatterGen model code is bundled in `mattergen/` — no separate package install needed. Never use system Python or create a new environment.

## Environment Setup

**Method 1 (recommended, ~2 min):**
```bash
bash scripts/uv_install.sh   # creates .venv and installs all deps (MatterGen bundled in mattergen/)
source .venv/bin/activate
```

**Method 2 (conda, >10 min):**
```bash
conda env create -f env.yml
conda activate matinvent
```

**FairChem/eSEN support** (for heat capacity rewards only) requires a separate conda environment — see `rewards/calculators/fairchem/README.md`.

## Running Experiments

```bash
# Basic RL run
python -u main.py expname=test pipeline=mat_invent model=mattergen reward=hhi logger=wandb

# Run in background (logs to exp_res/<expname>.log)
bash scripts/run_rl.sh

# Generate and evaluate structures
bash scripts/gen_eval.sh
```

Key Hydra parameters:
- `expname` — experiment name (output goes to `exp_res/<expname>/`)
- `pipeline` — `mat_invent` or `baseline`
- `model` — `mattergen` or `diffcsp`
- `reward` — any yaml name under `configs/reward/` (e.g. `hhi`, `band_gap`, `bulk_modulus`, `mag_den_hhi`)
- `logger` — `wandb` or `csv`
- `device` — `cuda`, `cuda:0`, `cpu`

To disable wandb: set `logger=csv`. Results are saved under `exp_res/<expname>/`.

## Architecture

### RL Loop (`pipeline/mat_invent.py:MatInvent.rl_step`)

Each RL step:
1. **Sample** — agent generates crystal batches → `invalid_filter` removes structurally invalid ones → optional `OptFilter` (validity/novelty/uniqueness/stability)
2. **Score** — `Reward.scoring` runs property calculators and linearly scales each property to [0,1]
3. **Memory update** — `LongTimeMem` accumulates all crystals; computes burden and diversity ratio metrics
4. **Diversity filter** (optional) — penalizes/zeroes rewards for over-represented compositions (Augmented Hill-Climb)
5. **Replay** — augments top-k samples with experience replay buffer
6. **Finetune** (`ft_step`) — REINFORCE-style loss on agent, with KL divergence regularization against the frozen prior model

The agent and prior are identical model instances loaded at startup; only the agent's weights are updated.

### Key Classes

| Class | File | Role |
|---|---|---|
| `MatInvent` | `pipeline/mat_invent.py` | Main RL orchestrator |
| `ReinL` | `pipeline/base.py` | Abstract RL base |
| `ModelSuite` | `models/suite/base.py` | Abstract model interface (load, sample, dataloader, save) |
| `MatterGenSuite` | `models/suite/mattergen.py` | MatterGen-specific implementation |
| `DiffCSPSuite` | `models/suite/diffcsp.py` | DiffCSP-specific implementation |
| `Reward` | `rewards/reward.py` | Combines property calculators, scales to [0,1], reduces multi-property to scalar |
| `Calculator` | `rewards/calculators/base.py` | Abstract calculator base |
| `LongTimeMem` | `memory/ltm.py` | Tracks all generated crystals; computes burden/diversity metrics |
| `ReplayBuffer` | `memory/replay_buffer.py` | Experience replay |
| `OptFilter` | `pipeline/filters/opt_filter.py` | Post-sample filter using MatterSim relaxation + reference dataset |

### Config System (Hydra)

`configs/base.yaml` is the root config. It composes sub-configs via the `defaults` list:
- `configs/pipeline/` — pipeline hyperparameters (RL epochs, topk_ratio, replay, diversity filter)
- `configs/model/` — model name, sample batch size, finetune LR/timesteps
- `configs/reward/` — property calculators, scaling range (`minv`/`maxv`), target direction (`ascending`/`descending`/float), threshold
- `configs/logger/` — wandb project settings or CSV path

Adding a new reward: create a new YAML in `configs/reward/` following the pattern in `configs/reward/hhi.yaml`, and implement a `Calculator` subclass under `rewards/calculators/`.

Adding a new model: subclass `ModelSuite` with `load_model`, `get_sampler`, `get_dataloader`, and `save_model`, then add a config under `configs/model/`.

### Property Calculators

| Calculator | Location | Properties |
|---|---|---|
| PyMatGen | `rewards/calculators/pymatgen/calc.py` | HHI, density, log abundance, price, MCIA, lattice mismatch |
| ALIGNN | `rewards/calculators/alignn/calc.py` | Band gap, bulk modulus, shear modulus, formation energy, dielectric, Pugh ratio |
| FairChem/eSEN | `rewards/calculators/fairchem/calc.py` | Heat capacity, elastic properties (requires separate env) |
| DFT | `rewards/calculators/dft/calc.py` | DFT-based properties via external job submission |
| SynScore | `rewards/calculators/syn_score/calc.py` | Synthesizability score (ensemble of 101 models in `model_pt/`) |
