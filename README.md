# DeepLearningFramework

작고 재현 가능한 딥러닝 연구 프레임워크입니다.

기본 구성:

- PyTorch
- Hydra
- TensorBoard
- pandas 기반 전처리
- CNN 기반 디노이징 baseline

## 시작

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train.py
```

TensorBoard:

```powershell
tensorboard --logdir logs
```

Hydra override 예시:

```powershell
python train.py train.epochs=5 train.batch_size=8 model.hidden_channels=32 seed=123
```

## 데이터셋

현재 baseline 은 `datasets/dataset_01` 부터 `datasets/dataset_10` 폴더를 스캔해 DataFrame 메타데이터를 만들고, 각 폴더를 하나의 소규모 데이터셋처럼 취급합니다.

실제 이미지 데이터가 아직 없더라도 baseline 검증이 가능하도록, 각 폴더 이름과 `sample_id` 를 기반으로 결정론적 synthetic denoising 샘플을 생성합니다.

향후 실제 파일 기반 학습으로 바꿀 때는 `src/data/preprocess.py` 와 `src/data/dataset.py` 를 확장하면 됩니다.
