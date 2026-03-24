# Фаза 5: Apparatus-Adversarial Training & Advanced Experiments

**Цель:** Обучить backbone извлекать apparatus-agnostic признаки. Исследовать fusion и современные backbone.

**Предусловие:** Фаза 4 завершена, лучшая модель Phase 4 выбрана как базовая.

---

## Идея (5.1 — DANN adversarial)

К backbone добавляется adversarial голова — классификатор снаряда (4 класса: Ball/Clubs/Hoop/Ribbon).
Backbone обучается одновременно:
1. **Хорошо** предсказывать элемент (основной loss)
2. **Плохо** предсказывать снаряд (adversarial loss через Gradient Reversal Layer)

В итоге backbone вырабатывает признаки, из которых нельзя определить снаряд → apparatus-agnostic.

```
Backbone → f_t
              ├── Temporal Head → element score   (основной loss: BCE)
              └── GRL → Apparatus Classifier      (adversarial loss: CrossEntropy, веса инвертированы)
```

---

## Задача 5.1: Реализация DANN

- [ ] **5.1.1** Реализовать `src/models/grl.py`:
  - `GradientReversalLayer` — в forward: identity; в backward: умножить градиент на `-lambda`
  - `lambda` растёт по schedule от 0 до 1 (как в оригинальном DANN)
- [ ] **5.1.2** Реализовать `src/models/apparatus_classifier.py`:
  - `ApparatusClassifier`: Linear(D→128) → ReLU → Linear(128→4)
  - Принимает усреднённый по времени backbone output за чанк
- [ ] **5.1.3** Модифицировать `src/scripts/train.py`: добавить режим `--adversarial`
  - Суммарный loss: `L = L_element + alpha * L_apparatus_adversarial`
  - `alpha` — гиперпараметр (поиск в HPO)
- [ ] **5.1.4** Добавить apparatus label в `src/data/frame_dataset.py` (из имени файла: Ball/Clubs/Hoop/Ribbon)

---

## Задача 5.2: Эксперименты DANN

- [ ] **5.2.1** Создать `configs/eff_b0_adversarial_bilstm_attn_opt.yaml`
  - Базовая архитектура: лучшая из Phase 4 + adversarial head
- [ ] **5.2.2** HPO 20 trials (добавить `alpha` и `grl_lambda_max` в пространство поиска)
- [ ] **5.2.3** Обучить seed=42
- [ ] **5.2.4** Запустить LOAO
- [ ] **5.2.5** Измерить FPS (adversarial head только при обучении, не при инференсе → FPS не меняется)

---

## Задача 5.3: Графики и выводы DANN

- [ ] **5.3.1** Добавить в scatter accuracy–speed из Phase 4
- [ ] **5.3.2** Добавить в LOAO heatmap
- [ ] **5.3.3** Обновить `docs/thesis_research_notes.md`
- [ ] **5.3.4** Коммит: "Phase 5: apparatus-adversarial training"

---

## Задача 5.4: Альтернативная гипотеза — Pose + Framediff fusion

**Мотивация из Phase 4:**

| Механизм | Ball | Clubs | Hoop | Ribbon | Слабое место |
|----------|:----:|:-----:|:----:|:------:|:------------:|
| eff_b0_framediff_bilstm_attn | 0.995 | **0.966** | 0.740 | 0.429 | Ribbon |
| pose_bilstm_attn | 0.989 | 0.655 | **1.000** | **1.000** | Clubs |

Pose закрывает Ribbon/Hoop через apparatus-agnostic скелет. Framediff даёт лучший Clubs (0.966) через motion channel. Комбинация может закрыть все 4 снаряда.

**Архитектура:**
```
Pose features [F, 99]     → Linear(99→H)  ─┐
Framediff features [F, D] → Linear(D→H)   ─┤→ concat → Linear(2H→H) → BiLSTM+Attn → logit
```

- [ ] **5.4.1** Реализовать `src/models/fusion_model.py` (два потока + concat + temporal head)
- [ ] **5.4.2** Реализовать `src/data/fusion_dataset.py` (синхронная загрузка pose + framediff кэшей)
- [ ] **5.4.3** Создать `configs/fusion_pose_framediff_bilstm_attn.yaml`
- [ ] **5.4.4** Патч `train.py` / `hpo.py` / `loao_cv.py` для `model_type: "fusion"`
- [ ] **5.4.5** HPO 20 trials
- [ ] **5.4.6** Train seed=42
- [ ] **5.4.7** LOAO — критерий: все 4 снаряда ≥ 0.70
- [ ] **5.4.8** FPS measurement

---

## Задача 5.5: Современный backbone — MobileNetV4

**Мотивация:** MobileNetV4 (Google, 2024) — специально спроектирован для edge/RT задач. Превосходит EfficientNet-B0 по ImageNet при сопоставимой скорости. Позволяет ответить на вопрос: улучшает ли современный backbone обобщаемость (LOAO) по снарядам?

| Backbone | Параметры | ImageNet top-1 | Год | Библиотека |
|----------|:---------:|:--------------:|:---:|:----------:|
| EfficientNet-B0 | 5.3M | 77.1% | 2019 | torchvision |
| MobileNetV4-Conv-Small | 3.8M | 73.8% | 2024 | timm |
| MobileNetV4-Conv-Medium | ~10M | 80.9% | 2024 | timm |

**Вариант для эксперимента:** `mobilenetv4_conv_medium` — лучший баланс accuracy/speed, превосходит eff_b0 по качеству признаков.

- [ ] **5.5.1** Установить `timm`: `pip install timm`; добавить в `requirements.txt`
- [ ] **5.5.2** Добавить `mobilenetv4_conv_medium` backbone в `src/models/backbone.py`:
  - `timm.create_model('mobilenetv4_conv_medium.e500_r256_in1k', pretrained=True, num_classes=0)`
  - `output_dim=960` (после global avg pool)
- [ ] **5.5.3** Извлечь фичи: `python src/scripts/extract_frame_features.py --backbone mobilenetv4_conv_medium --split all`
- [ ] **5.5.4** Создать конфиг `configs/mobilenetv4_medium_bilstm_attn.yaml`
- [ ] **5.5.5** HPO 20 trials
- [ ] **5.5.6** Train seed=42
- [ ] **5.5.7** LOAO — сравнить с eff_b0_bilstm_attn (0.739) и eff_b0_tsm_bilstm_attn (0.829)
- [ ] **5.5.8** FPS measurement — сравнить с eff_b0

**Ожидаемый результат:** mAP выше или на уровне eff_b0; LOAO — открытый вопрос.

---

## Критерии завершения Фазы 5

- [ ] Adversarial модель (5.1–5.3) обучена и оценена
- [ ] Fusion Pose+Framediff модель (5.4) обучена и оценена
- [ ] MobileNetV4 backbone (5.5) протестирован
- [ ] Лучший подход выбран как финальная модель
- [ ] Результаты добавлены в финальные графики
- [ ] `thesis_research_notes.md` обновлён с выводом
