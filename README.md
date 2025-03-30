# simon-lee-AI-personal-project

# 🧠 VGG16 모델 경량화 연구: L1-Norm 기반 필터 Pruning

본 프로젝트는 VGG16 기반 분류 모델에 대해 **필터 수준의 L1-norm 기반 Pruning 기법**을 적용하여  
모델을 경량화하고, 성능을 유지할 수 있는 범위를 실험적으로 탐구한 결과물입니다.

---

## 📂 프로젝트 구조

```
├── notebooks/              # 실험용 Colab/노트북 코드
├── pruning/                # 모델 pruning 및 분석 코드
│   ├── callback.py         # TimeHistory 콜백 정의
│   ├── conv_layer.py       # ConvLayer, CNNModel 클래스
│   └── pruning_utils.py    # L1-norm pruning, 시각화 함수 등
├── logs/                   # SQLite로 저장된 실험 로그 파일 (experiments.db 등)
├── results/                # 실험 결과 그래프, 보고서 스크립트
├── main_train.py           # 전체 학습/실험 실행 스크립트
├── save_sqlite.py          # 실험 기록을 SQLite에 저장하는 함수
└── README.md               # 본 파일
```

---

## 🧪 실험 개요

- **데이터셋**: Rice Image Dataset (75,000장, 5개 품종)
- **기반 모델**: Pre-trained **VGG16**
- **학습 방식**:
  - Feature Extraction (Conv Layer 고정)
  - Full Fine-Tuning (전체 레이어 학습)
- **Pruning 방법**: L1-norm 기반 필터 중요도 평가
- **Pruning 비율**: 5%, 10%, 20%
- **성능 복구 방식**: Classification Layer 재학습

---

## 🔍 핵심 결과 요약

| Pruning 비율 | Layer 재학습 여부 | Test Accuracy |
|--------------|-------------------|----------------|
| 0% (baseline) | ❌                | 99.54%         |
| 5%           | ✅                | 98.82%         |
| 10%          | ✅                | **99.06%**     |
| 20%          | ✅                | 98.68%         |

> ✅ Pruning 후 Classification Layer를 재학습하면,  
> **성능 저하 없이도 모델 경량화가 가능**함을 실험적으로 입증하였습니다.

---

## 💽 SQLite를 활용한 실험 기록 저장

- 각 실험의 학습 시간(`epoch_times`)과 메타데이터는 SQLite에 저장됨
- DB 파일: `experiments.db`
- 테이블: `training_logs`

```python
from save_sqlite import save_training_log_to_sqlite

save_training_log_to_sqlite(
    time_callback=time_callback,
    model_name="VGG16",
    pruning_ratio=0.1,
    test_accuracy=0.9906
)
```

---

## ▶️ 실행 예시

```python
# 학습
history = model.fit(..., callbacks=[TimeHistory()])

# 평가
acc = model.evaluate(test_data)[1]

# 로그 저장
save_training_log_to_sqlite(time_callback, "VGG16", 0.10, acc)
```

---

## 📊 향후 발전 방향

- Full Fine-Tuning 모델에 대한 pruning 실험 추가
- 레이어별 adaptive pruning 비율 도입
- 지식 증류(KD)와 결합한 경량화
- Streamlit 기반 실험 대시보드 연동

---

## 📚 참고 논문

- Han et al., “Learning both Weights and Connections…” (NIPS 2015)
- Molchanov et al., “Pruning CNNs for Resource Efficient Inference” (ICLR 2017)
- Liu et al., “Network Slimming” (ICCV 2017)
- Luo et al., “ThiNet” (ICCV 2017)

---

## ✍️ Author

> 본 실험은 Rice Image Dataset을 활용한 모델 경량화 연구 프로젝트의 일환으로 진행되었습니다.  
> 실험 기록, 시각화 및 분석은 모두 Colab + SQLite 기반으로 관리되었습니다.

---
