# AGENTS.md

이 문서는 이 저장소에서 작업하는 사람과 에이전트를 위한 공통 작업 규칙이다. 목표는 "가상환경 기반의 단순한 딥러닝 연구 시스템"을 빠르게 만들고, 실험 재현성과 유지보수성을 확보하는 것이다.

## 1. 프로젝트 목표

이 저장소는 다음 요구사항을 만족하는 최소한의 딥러닝 연구 프레임워크를 지향한다.

- Python 가상환경 기반으로 동작한다.
- `PyTorch + Hydra + TensorBoard` 조합을 기본으로 사용한다.
- 학습은 반드시 재현 가능해야 하며, seed 설정을 포함해야 한다.
- 구조는 최대한 단순해야 한다.
- 기본 모델은 CNN 기반의 기초적인 디노이징 블록을 사용한다.
- 전처리는 `pandas.DataFrame` 기반의 아주 간단한 기능만 제공한다.
- 데이터셋은 많지 않아도 되며, 최소 10개 정도의 폴더 단위 데이터로 학습 가능해야 한다.

## 2. 설계 원칙

- 과설계하지 않는다.
- 처음부터 분산학습, 복잡한 registry, 대규모 plugin 시스템을 넣지 않는다.
- 코드보다 설정이 많아지지 않도록 한다.
- 한 번에 이해 가능한 디렉터리 구조를 유지한다.
- 실험 실행, 로그 저장, 체크포인트 저장 경로는 명확해야 한다.

## 3. 권장 기술 스택

- Python 3.10 이상
- PyTorch
- Hydra
- TensorBoard
- pandas
- numpy

필요 시 추가 가능:

- torchvision
- tqdm
- pyyaml

## 4. 가상환경 규칙

항상 가상환경 안에서 실행한다.

권장 예시:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

규칙:

- 의존성은 가능한 한 `requirements.txt` 또는 `pyproject.toml` 하나로 관리한다.
- 로컬 환경 차이를 줄이기 위해 버전 고정 또는 최소 범위 지정이 필요하다.
- 실험 실행 전에 현재 Python 경로와 주요 패키지 버전을 확인할 수 있어야 한다.

## 5. 권장 디렉터리 구조

구조는 아래처럼 단순하게 유지한다.

```text
DeepLearningFramework/
  configs/
    config.yaml
    data/
    model/
    train/
  data/
    raw/
    processed/
  datasets/
    dataset_01/
    dataset_02/
    dataset_03/
    dataset_04/
    dataset_05/
    dataset_06/
    dataset_07/
    dataset_08/
    dataset_09/
    dataset_10/
  logs/
  outputs/
  src/
    data/
    models/
    training/
    utils/
  train.py
  requirements.txt
  AGENTS.md
  README.md
```

원칙:

- `datasets/` 아래에는 폴더 단위 데이터셋을 둔다.
- `data/raw` 는 원본 데이터, `data/processed` 는 전처리 결과를 둔다.
- `outputs/` 는 Hydra 실행 결과를 저장한다.
- `logs/` 는 TensorBoard 로그를 저장한다.
- `src/` 외부에는 핵심 로직을 흩뿌리지 않는다.

## 6. 데이터셋 규칙

데이터셋은 최소 10개 폴더를 기준으로 학습 가능해야 한다.

예시:

- `datasets/dataset_01`
- `datasets/dataset_02`
- `datasets/dataset_03`
- `...`
- `datasets/dataset_10`

규칙:

- 각 폴더는 동일한 형식의 샘플을 가져야 한다.
- 폴더명은 규칙적으로 유지한다.
- train/val split 이 필요하면 코드에서 분리하거나 메타 파일로 관리한다.
- 샘플 수가 적어도 실행 가능해야 한다.
- 경로 하드코딩을 피하고 Hydra 설정으로 데이터 루트를 받는다.

## 7. 전처리 규칙

전처리는 "아주 간단한 DataFrame 기반 처리"만 우선 지원한다.

권장 범위:

- 파일 목록 수집
- 경로 매핑
- 라벨 또는 메타정보 정리
- train/val split 생성
- csv 저장

권장 방식:

- `pandas.DataFrame` 하나로 샘플 메타데이터를 관리한다.
- 전처리 결과는 csv 또는 parquet 중 하나로 단순 저장한다.
- 전처리 단계에서 복잡한 feature engineering 은 하지 않는다.

예시 컬럼:

- `input_path`
- `target_path`
- `split`
- `dataset_name`
- `sample_id`

## 8. 모델 규칙

기본 모델은 CNN 기반의 매우 단순한 디노이징 블록으로 시작한다.

권장 예시:

- `Conv2d -> ReLU -> Conv2d -> ReLU -> Conv2d`
- residual connection 가능
- batch norm 은 꼭 필요할 때만 추가

원칙:

- 처음부터 U-Net 전체 구조를 강제하지 않는다.
- 블록 수와 채널 수는 Hydra 설정으로 바꿀 수 있게 한다.
- baseline 은 작고 빠르게 돌아야 한다.
- 모델 이름은 명확하게 유지한다. 예: `SimpleDenoiser`, `BasicDenoiseBlock`

## 9. 학습 재현성 규칙

재현성은 필수다.

반드시 포함할 항목:

- Python random seed
- numpy seed
- torch seed
- torch cuda seed
- DataLoader worker seed
- 설정값 저장
- 실행 시점의 주요 하이퍼파라미터 로그 저장

권장 구현:

```python
import os
import random
import numpy as np
import torch

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

추가 규칙:

- seed 는 Hydra config 에 반드시 존재해야 한다.
- 실행마다 최종 config snapshot 을 저장한다.
- 결과 폴더명에 날짜만 쓰지 말고 run name 또는 seed 를 포함하는 것을 권장한다.

## 10. Hydra 사용 규칙

Hydra 는 복잡하게 쓰지 말고, 아래 범위에서만 단순하게 사용한다.

- 데이터 설정
- 모델 설정
- 학습 설정
- 경로 설정
- seed 설정

권장 예시:

- `configs/config.yaml`
- `configs/model/basic_denoiser.yaml`
- `configs/data/default.yaml`
- `configs/train/default.yaml`

규칙:

- config group 수를 과도하게 늘리지 않는다.
- 하나의 실험을 이해하기 위해 많은 yaml 파일을 열어야 하는 구조를 피한다.
- CLI override 는 간단히 유지한다.

예시:

```powershell
python train.py model=basic_denoiser train.epochs=20 seed=42
```

## 11. TensorBoard 규칙

TensorBoard 는 기본 로깅 도구로 사용한다.

반드시 기록할 항목:

- train loss
- validation loss
- learning rate
- 예측 결과 샘플 이미지 가능 시 저장

규칙:

- 로그 디렉터리는 일관된 구조를 가진다.
- Hydra output 과 TensorBoard log 경로를 혼동하지 않는다.
- 실행 후 바로 `tensorboard --logdir logs` 로 확인 가능해야 한다.

## 12. 학습 스크립트 규칙

엔트리포인트는 최대한 단순하게 유지한다.

권장 흐름:

1. Hydra config 로드
2. seed 고정
3. dataset/dataframe 로드
4. dataloader 생성
5. model 생성
6. optimizer/loss 생성
7. train/validation loop 실행
8. checkpoint 및 logs 저장

규칙:

- `train.py` 에 모든 세부 구현을 몰아넣지 않는다.
- 하지만 작은 프로젝트인 만큼 지나친 추상화도 피한다.
- 한 파일이 너무 커지면 `data`, `models`, `training`, `utils` 로만 나눈다.

## 13. 체크포인트 및 출력 규칙

- 최고 성능 체크포인트 1개 이상 저장
- 마지막 epoch 체크포인트 저장
- 사용한 config 저장
- 가능하면 metrics 요약 json 또는 csv 저장

권장 출력:

- `outputs/.../config.yaml`
- `outputs/.../metrics.csv`
- `outputs/.../checkpoints/best.pt`
- `outputs/.../checkpoints/last.pt`

## 14. 코딩 규칙

- 함수와 클래스 이름은 의미가 분명해야 한다.
- 한 함수가 여러 책임을 갖지 않도록 한다.
- 주석은 적게, 대신 이름을 명확하게 짓는다.
- 타입 힌트는 가능한 범위에서 사용한다.
- 실험 코드라도 기본적인 오류 처리는 포함한다.
- 경로는 `pathlib.Path` 사용을 권장한다.

## 15. 우선 구현 순서

이 저장소에서 작업할 때는 아래 순서를 우선한다.

1. 가상환경 및 의존성 정의
2. Hydra 기본 config 작성
3. seed 유틸 작성
4. DataFrame 기반 메타데이터 전처리 작성
5. dataset/dataloader 작성
6. CNN 기반 디노이징 baseline 모델 작성
7. train/validation loop 작성
8. TensorBoard logging 연결
9. checkpoint 저장
10. 샘플 데이터셋 10개 폴더 기준 실행 검증

## 16. 에이전트 작업 규칙

이 저장소에서 자동화 에이전트가 작업할 때는 다음을 따른다.

- 단순한 구조를 해치지 않는 방향으로 수정한다.
- 새 기능을 넣을 때 먼저 baseline 을 깨지 않는지 확인한다.
- 재현성과 경로 일관성을 최우선으로 둔다.
- Hydra 설정과 코드 기본값이 충돌하지 않게 한다.
- 불필요한 추상화, 과도한 설정 분리, 과한 디자인 패턴 도입을 피한다.
- 사용자가 요청하지 않으면 무거운 프레임워크를 추가하지 않는다.

## 17. 완료 기준

최소 완료 기준은 아래와 같다.

- 가상환경에서 설치 가능
- `train.py` 한 번 실행으로 학습 시작 가능
- Hydra override 동작
- TensorBoard 로그 생성
- seed 포함 재현성 설정 존재
- 10개 폴더 기반의 간단한 데이터셋 입력 지원
- CNN 기반 디노이징 baseline 동작
- DataFrame 기반 전처리 기능 존재

이 문서의 핵심은 "작지만 재현 가능하고, 바로 실험 가능한 딥러닝 연구 시스템"이다.
