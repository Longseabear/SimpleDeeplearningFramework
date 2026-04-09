# DeepLearningFramework

간단하지만 재현 가능한 MNIST 디노이징 학습 프레임워크입니다.

기본 구성:

- PyTorch
- torchvision
- Hydra
- TensorBoard
- pandas 기반 metadata DataFrame
- 3개의 convolution + 3개의 fully connected layer로 구성한 교육용 baseline

## 시작

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

첫 실행 시 `torchvision.datasets.MNIST`로 데이터를 내려받고, `datasets/mnist_png` 아래에 PNG로 저장합니다. 이후 metadata DataFrame을 만들고, 이를 섞어서 `train/validation/test` split을 구성합니다.

TensorBoard:

```powershell
tensorboard --logdir logs
```

Hydra override 예시:

```powershell
python train.py train.epochs=3 train.batch_size=64 data.noise_std=0.3 seed=123
```

체크포인트:

- 매 epoch마다 `outputs/.../checkpoints/last.pt` 저장
- 가장 좋은 validation loss는 `outputs/.../checkpoints/best.pt` 저장
- 이어서 학습할 때는 `train.resume_from_checkpoint` 사용

```powershell
python train.py train.resume_from_checkpoint="outputs/2026-04-09/11-00-00/checkpoints/last.pt" train.epochs=10
```

데이터와 출력:

- `datasets/`: MNIST PNG 이미지 10개 폴더로 분산 저장
- `datasets/mnist_png/`: MNIST PNG 이미지 저장
- `data/processed/metadata.csv`: `train/validation/test` split이 포함된 DataFrame 메타데이터
- `logs/`: TensorBoard 로그
- `outputs/...`: Hydra config snapshot, metrics, checkpoint, runtime 환경 정보
