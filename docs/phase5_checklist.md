# Фаза 5: MobileNetV4 Backbone

**Цель:** Проверить, улучшает ли современный backbone (MobileNetV4, Google 2024) обобщаемость по снарядам (LOAO) по сравнению с EfficientNet-B0.

**Предусловие:** Фаза 4 завершена, лучшая модель Phase 4 выбрана как baseline.

---

## Мотивация

MobileNetV4 (Google, 2024) — специально спроектирован для edge/RT задач. Превосходит EfficientNet-B0 по ImageNet при сопоставимой скорости.

| Backbone | Параметры | ImageNet top-1 | Год | Библиотека |
|----------|:---------:|:--------------:|:---:|:----------:|
| EfficientNet-B0 | 5.3M | 77.1% | 2019 | torchvision |
| MobileNetV4-Conv-Small | 2.5M | 73.8% | 2024 | timm |

**Вариант для эксперимента:** `mobilenetv4_conv_small` — лучший баланс accuracy/speed, превосходит eff_b0 по качеству признаков.

---

## Задача 5.1: Подготовка backbone

- [x] **5.1.1** Установить `timm`: добавлен в `requirements.txt`
- [x] **5.1.2** Добавить `mobilenetv4_conv_small` backbone в `src/models/backbone.py`:
  - `timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=0)`
  - `output_dim=1280` (после global avg pool)
- [x] **5.1.3** Фичи извлечены: train=200, valid=68, test=68 (336 файлов в `data/frame_features/`)

---

## Задача 5.2: Архитектуры (аналогично eff_b0: BiLSTM+Attn, BiLSTM, TCN)

### 5.2.A: mv4_small + BiLSTM+Attention

- [x] **5.2.A.1** HPO завершён — 22 trials, **best mAP@0.5 = 0.8945** (trial #15)
  - Best params: lr=0.000565, dropout=0.313, chunk_size=261, hidden_dim=64, n_layers=3
- [x] **5.2.A.2** Multi-seed обучение (42, 123, 2024)
  - seed=42 → 0.894 | seed=123 → 0.796 | seed=2024 → 0.873
  - **Mean: 0.854 ± 0.042**
- [x] **5.2.A.3** Evaluate test (seed=42): mAP@0.5=0.873, P=0.894, R=0.937, BE=0.235s
- [x] **5.2.A.4** LOAO (seed=42) — **FAIL** (Ball=0.530, Clubs=0.818, Hoop=0.660, Ribbon=0.170, Mean=0.545±0.239)

### 5.2.B: mv4_small + BiLSTM

- [x] **5.2.B.1** HPO завершён — **best mAP@0.5 = 0.8896** (trial #0)
  - Best params: lr=0.000433, dropout=0.393, chunk_size=358, hidden_dim=64, n_layers=3
- [x] **5.2.B.2** Multi-seed обучение (42, 123, 2024)
  - seed=42 → 0.890 | seed=123 → 0.801 | seed=2024 → 0.886
  - **Mean: 0.859 ± 0.041**
- [ ] **5.2.B.3** Evaluate test
- [ ] **5.2.B.4** LOAO

### 5.2.C: mv4_small + TCN

- [x] **5.2.C.1** HPO завершён — **best mAP@0.5 = 0.7314** (trial #9, 10 complete)
  - Best params: lr=0.000329, dropout=0.374, chunk_size=297
- [ ] **5.2.C.2** Multi-seed обучение (42, 123, 2024) — в процессе
- [ ] **5.2.C.3** Evaluate test
- [ ] **5.2.C.4** LOAO

---

## Задача 5.3: Оценка

- [ ] **5.3.1** LOAO — критерий: все 4 снаряда ≥ 0.70
  - Сравнить с eff_b0_bilstm_attn (LOAO=0.739) и pose_causal_tcn (LOAO=0.968)
- [ ] **5.3.2** FPS benchmark (e2e: decode + MV4 GPU + temporal):
  - `python src/scripts/measure_fps_e2e.py` — добавить mv4_medium в список
  - Сравнить с eff_b0 (~369 FPS)
- [ ] **5.3.3** Записать результаты в `docs/comparison_table.md`

---

## Задача 5.4: Графики и выводы

- [ ] **5.4.1** Добавить точку MV4 в scatter accuracy–speed (fig14)
- [ ] **5.4.2** Добавить строку в LOAO heatmap (fig15)
- [ ] **5.4.3** Обновить `docs/thesis_research_notes.md`
- [ ] **5.4.4** Коммит: "Phase 5: MobileNetV4 backbone evaluation"

---

## Финальные метрики (заполнить)

| Метрика | eff_b0_bilstm_attn | eff_b0_bilstm | eff_b0_tcn | mv4_bilstm_attn | mv4_bilstm | mv4_tcn |
|---------|--------------------|---------------|------------|-----------------|------------|---------|
| mAP@0.5 (valid, mean) | 0.856±0.008 | — | — | **0.854±0.042** | | |
| mAP@0.5 (test, seed=42) | 0.788 | 0.791 | 0.900 | **0.873** | | |
| LOAO Mean | 0.739 | — | — | **0.545** ❌ | | |
| LOAO Ribbon | 0.800 | — | — | **0.170** ❌ | | |
| BE (test) | — | — | — | **0.235s** | | |
| FPS (e2e) | 358 | 368 | 369 | | | |
| Размер модели | 26.5 MB | — | — | **14.7 MB** | | |

---

## Критерии завершения Фазы 5

- [x] `mobilenetv4_conv_small` backbone добавлен и проверен
- [x] Фичи извлечены для всех 3 сплитов (336 видео)
- [x] HPO завершён (22 trials, best=0.8945)
- [ ] 3 seeds обучены, метрики зафиксированы
- [ ] LOAO завершён (все 4 снаряда)
- [ ] FPS измерен (e2e)
- [ ] Результаты добавлены в графики и `thesis_research_notes.md`
- [ ] Коммит: "Phase 5: MobileNetV4 backbone evaluation"
