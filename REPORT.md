# MNIST 숫자 인식 과제 보고서

## 0. 반/팀원

| 항목 | 내용 |
| --- | --- |
| 반 | (예: AI 1반) |
| 팀명 | (예: 팀명 입력) |
| 팀원 | (예: 홍길동, 김철수) |

---

## 1. 실험 목적

본 프로젝트는 PyTorch, TensorFlow 같은 딥러닝 프레임워크를 사용하지 않고, NumPy만으로 MNIST 손글씨 숫자 분류용 신경망을 직접 구현하는 것을 목표로 하였다.  
또한 단순히 정확도를 얻는 것에 그치지 않고, `forward -> loss -> backward -> optimizer update`의 전체 학습 흐름을 직접 구현하고 이해하는 데 목적이 있다.

---

## 2. 모델 구조

본 프로젝트에서는 MNIST 손글씨 숫자 분류를 위해 다층 퍼셉트론(MLP) 구조의 신경망을 사용하였다.  
입력 이미지는 `28 x 28` 크기의 흑백 이미지이며, 이를 1차원 벡터로 펼쳐 총 `784`차원의 입력으로 사용하였다.

전체 모델 구조는 다음과 같다.

`784 -> Affine(512) -> BatchNorm -> ReLU -> Dropout -> Affine(256) -> BatchNorm -> ReLU -> Dropout -> Affine(10) -> Softmax`

| 구분 | 내용 |
| --- | --- |
| 입력층 | 784차원 (`28 x 28` 이미지를 펼친 벡터, `0~1` 정규화) |
| 은닉층 1 | `Affine(512) -> BatchNorm -> ReLU -> Dropout` |
| 은닉층 2 | `Affine(256) -> BatchNorm -> ReLU -> Dropout` |
| 출력층 | `Affine(10) -> Softmax` |

각 레이어의 역할은 다음과 같다.

- `Affine`: 입력에 가중치와 편향을 적용하는 완전연결층
- `BatchNorm`: 은닉층 출력 분포를 정규화하여 학습을 안정화
- `ReLU`: 음수 값을 0으로 바꾸어 비선형성 부여
- `Dropout`: 학습 중 일부 뉴런을 무작위로 비활성화하여 과적합 감소
- `Softmax`: 최종 점수를 클래스별 확률 형태로 변환

---

## 3. 학습 설정

### CASE 1

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | Adam |
| 학습률 (lr) | 0.001 |
| epochs | 20 |
| batch_size | 128 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He (bias 0) |

### CASE 2

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | Adam |
| 학습률 (lr) | 0.001 |
| epochs | 20 |
| batch_size | 128 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He (bias 0) |

### CASE 3

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | Adam |
| 학습률 (lr) | 0.001 |
| epochs | 20 |
| batch_size | 128 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He (bias 0) |

### CASE 4

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | Adam |
| 학습률 (lr) | 0.001 |
| epochs | 20 |
| batch_size | 128 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He (bias 0) |

### CASE 5

| 항목 | 값 |
| --- | --- |
| 옵티마이저 | SGD |
| 학습률 (lr) | 0.001 |
| epochs | 20 |
| batch_size | 128 |
| Dropout 비율 | 0.5 |
| BatchNorm momentum | 0.9 |
| 가중치 초기화 | He (bias 0) |

---

## 4. 실험 환경

실험은 Python 3.11 환경에서 진행하였다.  
신경망 구현에는 외부 딥러닝 프레임워크를 사용하지 않고, NumPy만을 이용해 `forward`, `backward`, `optimizer update`를 직접 구현하였다.

| 항목 | 내용 |
| --- | --- |
| Python | 3.11 |
| 주요 라이브러리 | NumPy, Matplotlib |
| 실행 환경 | 로컬 환경 (노트북 gpu) |
| 학습 시간 | (실험 후 입력) |

---

## 5. 결과

실험 결과는 아래 표에 정리하였다.

| Case | Optimizer | Learning Rate | Dropout Ratio | Epochs | Batch Size | Test Accuracy | Training Time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Case 1 | Adam | 0.001 | 0.2 | 15 | 128 | 98.37% | 4m 29s |
| Case 2 | Adam | 0.0005 | 0.2 | 15 | 128 | 98.18% | 4m 28s |
| Case 3 | Adam | 0.0001 | 0.3 | 15 | 128 | 97.65% | 5m 50s |
| Case 4 | Adam | 0.001 | 0.5 | 20 | 128 | 98.59% | 6m 3s |
| Case 5 | SGD | 0.001 | 0.5 | 20 | 128 | 91.85% | 2m 35s |


### 최종 결과 요약

| 항목 | 값 |
| --- | --- |
| 최종 Test Accuracy |  |
| 총 파라미터 수 |  |
| 최종 선택 설정 |  |

### 손실 곡선

- `plot_loss_history(loss_history)`로 그래프를 확인하고 첨부하거나, 아래처럼 요약하여 기록한다.
- 예시:
  - `Epoch 1 Loss: ...`
  - `Epoch 5 Loss: ...`
  - `Epoch 10 Loss: ...`
  - `Epoch 15 Loss: ...`

---

## 6. 회고

이번 실험에서는 NumPy만으로 신경망의 전체 학습 흐름을 직접 구현하면서, 각 단계가 어떻게 연결되는지를 확인할 수 있었다.  
특히 `forward -> loss -> backward -> optimizer update` 순서가 실제 학습에서 어떻게 동작하는지 명확히 이해할 수 있었다.

실험 과정에서 확인한 주요 사항은 다음과 같다.

- learning rate에 따라 loss 감소 속도와 학습 안정성이 달라졌다.
- dropout ratio를 조정하면서 과적합 방지와 test accuracy 사이의 균형을 비교할 수 있었다.
- epoch 수를 늘렸을 때 추가 학습이 성능 향상으로 이어지는지 확인할 수 있었다.
- BatchNorm과 Dropout은 모델의 일반화 성능과 학습 안정성에 도움을 주었다.

최종적으로는 가장 안정적으로 loss가 감소하고, 가장 높은 test accuracy를 기록한 설정을 선택하였다.  
이 과정을 통해 단순히 정확도를 얻는 것뿐 아니라, 신경망 내부 구성 요소가 어떤 역할을 하는지 직접 구현하며 이해할 수 있었다.