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

## Step By Step Setup On A New Windows PC

### 1. Clone the repository

```powershell
git clone https://github.com/Longseabear/SimpleDeeplearningFramework.git
cd SimpleDeeplearningFramework
```

### 2. If PowerShell script execution is blocked

If `Activate.ps1` or `scripts/*.ps1` fails because of execution policy, run one of these first.

Current terminal session only:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 3. Create the virtual environment

```powershell
python -m venv .venv
```

### 4. Activate the virtual environment

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Python packages

```powershell
pip install -r requirements.txt
```

### 6. Prepare the dataset

This downloads MNIST, exports PNG files, and creates `metadata.csv`.

```powershell
python .\scripts\prepare_data.py
```

### 7. Start training

```powershell
python .\train.py
```

## Recommended Shortcut After Installation

Once `.venv` already exists and packages are installed, you can use the helper script:

```powershell
.\scripts\train.ps1
```

This will:

1. prepare MNIST PNG files and metadata
2. start training with the current Hydra config

## Common Commands

Prepare dataset only:

```powershell
python .\scripts\prepare_data.py
```

Train with default config:

```powershell
python .\train.py
```

Train with Hydra overrides:

```powershell
.\scripts\train.ps1 train.epochs=3 train.batch_size=64 data.noise_std=0.3 seed=123
```

Install everything from scratch with helper script:

```powershell
.\scripts\setup.ps1
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
python -m tensorboard.main --logdir logs
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
- `torchinfo>=1.8,<2.0`
