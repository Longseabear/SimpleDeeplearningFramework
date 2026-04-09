# DeepLearningFramework

Simple, reproducible MNIST denoising training framework for study and small experiments.

Core stack:

- PyTorch
- torchvision
- Hydra
- TensorBoard
- pandas metadata DataFrame
- Simple baseline model with 3 convolution layers and 3 fully connected layers

## Project Layout

- `train.py`: training entry point only
- `src/data/`: MNIST download, PNG export, metadata, dataset
- `src/models/`: denoising model
- `src/training/`: train/validation loop, checkpointing, TensorBoard
- `configs/`: Hydra config files
- `scripts/`: setup, data prep, train, and reproduce helpers

## Quick Start On A New PC

Windows PowerShell:

```powershell
git clone https://github.com/Longseabear/SimpleDeeplearningFramework.git
cd SimpleDeeplearningFramework
.\scripts\setup.ps1
.\scripts\train.ps1
```

What happens:

1. `scripts/setup.ps1` creates `.venv` and installs `requirements.txt`
2. `scripts/prepare_data.py` downloads MNIST and exports PNG files to `datasets/mnist_png/`
3. `metadata.csv` is created at `data/processed/metadata.csv`
4. `train.py` starts training with Hydra, TensorBoard, and checkpoints

## Manual Commands

Create environment and install packages:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Prepare dataset only:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_data.py
```

Train with default config:

```powershell
.\.venv\Scripts\python.exe .\train.py
```

Train with Hydra overrides:

```powershell
.\scripts\train.ps1 train.epochs=3 train.batch_size=64 data.noise_std=0.3 seed=123
```

## Reproducing A Previous Run

Hydra saves the exact config used for each run under `outputs/<date>/<time>/config.yaml`.

To rerun the same experiment from that saved config:

```powershell
.\scripts\reproduce_run.ps1 .\outputs\2026-04-09\13-30-25
```

You can still apply extra overrides on top:

```powershell
.\scripts\reproduce_run.ps1 .\outputs\2026-04-09\13-30-25 train.epochs=10
```

To resume from a checkpoint instead of restarting from scratch:

```powershell
.\scripts\train.ps1 train.resume_from_checkpoint="outputs\2026-04-09\13-30-25\checkpoints\last.pt" train.epochs=10
```

## Outputs

Hydra run outputs:

- `outputs/.../config.yaml`
- `outputs/.../metadata_snapshot.csv`
- `outputs/.../metrics.csv`
- `outputs/.../runtime_env.txt`
- `outputs/.../train.log`
- `outputs/.../checkpoints/last.pt`
- `outputs/.../checkpoints/best.pt`

TensorBoard logs:

- `logs/<run_name>_seed<seed>/`

MNIST data:

- `data/raw/`: original downloaded MNIST files
- `datasets/mnist_png/`: exported PNG images
- `data/processed/metadata.csv`: shuffled `train/validation/test` metadata

## TensorBoard

```powershell
.\.venv\Scripts\python.exe -m tensorboard.main --logdir logs
```

Then open:

```text
http://localhost:6006
```

## Reproducibility Notes

- Seed is controlled through Hydra config
- Python, NumPy, PyTorch, CUDA, and DataLoader worker seeding are applied
- Runtime environment information is saved to `runtime_env.txt`
- Final config snapshot is saved for every run

## Requirements

Current project dependencies:

- `torch>=2.2,<3.0`
- `torchvision>=0.17,<1.0`
- `hydra-core>=1.3,<2.0`
- `tensorboard>=2.16,<3.0`
- `pandas>=2.0,<3.0`
- `numpy>=1.26,<3.0`
- `tqdm>=4.66,<5.0`
