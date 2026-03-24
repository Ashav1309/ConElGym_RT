# Фаза 4: Generalization Ablation Study

**Цель:** Исследовать причины низкого LOAO у 2D моделей и найти архитектурные решения,
сохраняющие RT-требования. Построить графики accuracy–speed и LOAO heatmap для диссертации.

**Предусловие:** Фаза 3 завершена (LOAO для топ-моделей готов).

**Протокол:** seed=42, HPO=20 trials, LOAO обязателен, FPS измеряется.

**Модели для сравнения (топ-6 из Phase 3, исключены 2 слабейших):**
- `eff_b0_bilstm_attn` (0.830±0.031) — финальная модель
- `eff_b0_bilstm` (0.827±0.031)
- `mv3_bilstm` (0.768±0.004)
- `eff_b0_tcn` (0.763±0.098)
- `mv3_bilstm_attn` (0.724±0.027)
- `mv3_tcn` (0.490±0.053)
- ~~`eff_b0_mlp` (0.159±0.016)~~ — исключена
- ~~`mv3_small` (0.045±0.038)~~ — исключена

---

## Задача 4.0: Завершить LOAO Фазы 3

- [x] **4.0.1** LOAO для всех 8 моделей завершён (результаты в `docs/phase3_checklist.md`)
- [x] **4.0.2** Результаты внесены в `docs/phase3_checklist.md` и `docs/thesis_research_notes.md`
- [x] **4.0.3** LOAO запущен для всех 8 моделей ✅

---

## Задача 4.1: Baseline — Accuracy vs Speed график (2D + S3D)

- [x] **4.1.1** Данные из Фазы 3 собраны (все 8 моделей в `generate_plots.py`: MODELS_SEED42, LOAO)
- [x] **4.1.2** S3D референс v2 добавлен: mAP@0.5=0.993, LOAO=0.879 (в Fig 10, Fig 11); FPS не измерен → точка убрана из scatter plot (FPS≈5 был оценочным — удалено 2026-03-20)
- [x] **4.1.3** Scatter plot построен: `data/plots/results/fig4_fps_vs_map.png` (FPS log, RT zone, S3D ref)
- [x] **4.1.4** LOAO heatmap построен: `data/plots/results/fig10_loao_heatmap.png` (8 моделей × 4 снаряда)

---

## Задача 4.2: Архитектура A — 3D backbone (S3D RT-reference)

**Цель:** Показать, что 3D backbone решает проблему LOAO, но нарушает RT-требование.

- [x] **4.2.1** Создан `configs/s3d_rt_reference.yaml` ✅ (2026-03-23)
- [x] **4.2.2** Реализован `src/models/backbone_s3d.py` ✅ (2026-03-23)
  - `S3DBackbone`: S3D (Kinetics-400), заморожен, [B,C,T,H,W] → [B,1024]
  - `extract_s3d_features_rt`: скользящее окно, возвращает FPS
  - `src/scripts/measure_s3d_fps.py`: замер FPS (прогрев + N прогонов)
- [x] **4.2.3** FPS S3D на Ball_001.mp4: **261.9 FPS** (runs: 235.7 / 269.3 / 280.5) ✅ (2026-03-23)
  - ⚠️ Это включает декодирование видео + S3D CNN. 2D FPS (9K–13K) — только LSTM без backbone → сравнение нечестное
  - S3D: 30.2 MB, 7.9M параметров; требует 16-кадровый буфер (640ms latency @25fps)
  - Vs eff_b0_bilstm_attn: 5.3 MB, 1.4M параметров
- [x] **4.2.4** LOAO — используем результат из Phase 3/4: LOAO=0.879±0.016 (нет смысла повторять, признаки те же) ✅
- [x] **4.2.5** Внести FPS в графики (после 4.2.3) — S3D e2e FPS=262±22 внесён в fig14 scatter ✅ (2026-03-24)

**Ожидаемый результат:** LOAO ~0.85+, FPS <10 → «3D = теряем скорость».

---

## Задача 4.3: Архитектура B — Frame Difference Channel

**Цель:** Добавить motion signal к 2D backbone через покадровую разность. Проверить на всех 6 топ-моделях Phase 3.

### 4.3 — EfficientNet-B0 + framediff (пилот)

- [x] **4.3.1** `src/models/backbone.py`: добавлен `efficientnet_b0_framediff`
  - Входной тензор: [I_t | I_t − I_{t-1}] = 6 каналов
  - Первый conv: `in_channels=6`; pretrained веса для каналов 0–2, нули для 3–5
  - Для первого кадра: diff = 0
- [x] **4.3.2** Создан `configs/efficientnet_b0_framediff_bilstm_attn_opt.yaml`
  - Стартовые HP идентичны лучшей модели `eff_b0_bilstm_attn_opt`
- [x] **4.3.2b** `extract_frame_features.py` поддерживает frame diff (последовательная обработка с буфером)
- [x] **4.3.2c** Извлечение фич завершено: 336 видео (200+68+68), 0 ошибок ✅
- [x] **4.3.3** HPO 20 trials: best mAP@0.5=0.894 ✅
- [x] **4.3.4** Обучить seed=42 — best mAP@0.5=0.894 (valid), BE=0.25s, Recall=0.905 ✅
- [x] **4.3.5** LOAO: Ball=0.995 ✅ Clubs=0.966 ✅ Hoop=0.740 ✅ Ribbon=0.429 ❌ | Mean=0.783±0.226 (Ribbon провален; хуже TSM: 0.783 vs 0.829)
- [x] **4.3.6** FPS=9,026 (9.0K) | mAP@0.5=0.777 (test) | BE=0.26s | Recall=0.889 ✅ (2026-03-21, compare_models.py, test split)
- [x] **4.3.7** Внесено в generate_plots.py: PHASE5_MODELS, PHASE5_LOAO, fig12_phase5_loao, fig13_phase5_fps_map ✅

### 4.3 — Расширение framediff на все 6 топ-моделей

- [x] **4.3.8** Реализован `mobilenet_v3_small_framediff` backbone в `src/models/backbone.py` ✅
  - Аналогично `efficientnet_b0_framediff`: вход 6 каналов, pretrained для 0–2, нули для 3–5
- [x] **4.3.9** Извлечены фичи для `mv3_framediff`: 200+68+68=336 видео, 0 ошибок ✅ (2026-03-22)
- [x] **4.3.10** `eff_b0_bilstm_framediff` (`efficientnet_b0_framediff_bilstm`): HPO best=0.905 | train mAP=0.905 | LOAO: Ball=0.898 ✅ Clubs=0.893 ✅ Hoop=0.603 ❌ Ribbon=0.818 ✅ | Mean=0.803±0.120 ❌ Hoop (2026-03-22)
- [x] **4.3.11** `mv3_bilstm_framediff` (`mobilenet_v3_small_framediff_bilstm`): HPO best=0.779 | train mAP=0.779 | LOAO: Ball=0.621 ❌ Clubs=0.636 ❌ Hoop=0.420 ❌ Ribbon=0.403 ❌ | Mean=0.520±0.109 ❌ (2026-03-22)
- [x] **4.3.12** `eff_b0_tcn_framediff` (`efficientnet_b0_framediff_tcn`): HPO best=0.793 | train mAP=0.555 | LOAO: Ball=0.182 ❌ Clubs=0.545 ❌ Hoop=0.432 ❌ Ribbon=0.549 ❌ | Mean=0.427±0.149 ❌ все (2026-03-23)
- [x] **4.3.13** `mv3_bilstm_attn_framediff` (`mobilenet_v3_small_framediff_bilstm_attn`): HPO best=0.793 | train mAP=0.793 | LOAO: Ball=0.545 ❌ Clubs=0.628 ❌ Hoop=0.534 ❌ Ribbon=0.284 ❌ | Mean=0.498±0.129 ❌ (2026-03-22)
- [ ] ~~**4.3.14**~~ `mv3_tcn_framediff`: **ПРИОСТАНОВЛЕН** — фокус на eff_b0

---

## Задача 4.4: Архитектура C — TSM (Temporal Shift Module)

**Цель:** Дать 2D backbone темпоральный контекст без увеличения latency. Проверить на всех 6 топ-моделях Phase 3.

### 4.4 — EfficientNet-B0 + TSM (пилот)

- [x] **4.4.1** Реализован `src/models/tsm.py`:
  - `TemporalShift` — сдвиг 1/8 каналов: влево (t-1) + вправо (t+1) в bidirectional режиме
  - Каузальный режим (стриминг): только левый сдвиг + буфер
  - Все MBConv блоки EfficientNet-B0 (`features[1..7]`) обёрнуты в `TemporalShift`
  - Параметры идентичны базовому eff_b0 (TSM не добавляет параметров)
- [x] **4.4.1b** `backbone.py`: добавлен `efficientnet_b0_tsm`; `extract_frame_features.py`: tsm_mode (весь видео [T,C,H,W] одним проходом)
- [x] **4.4.2** Создан `configs/efficientnet_b0_tsm_bilstm_attn_opt.yaml`
- [x] **4.4.2b** Извлечение TSM фич завершено: 336 видео (200+68+68), 0 ошибок ✅ (фикс OOM: чанки по 64 кадра с 1-кадровым overlap)
- [x] **4.4.3** HPO 20 trials: best mAP@0.5=0.896 ✅
- [x] **4.4.4** Обучить seed=42 — best mAP@0.5=0.896 (valid), BE=0.21s, Recall=0.952 ✅
- [x] **4.4.5** LOAO: Ball=1.000 ✅ Clubs=0.790 ✅ Hoop=0.903 ✅ Ribbon=0.621 ❌ | Mean=0.829±0.141 (Hoop исправлен! +0.267 vs Phase3, Ribbon упал)
- [x] **4.4.6** FPS=13,131 (13.1K) | mAP@0.5=0.886 (test) | BE=0.25s | Recall=0.921 ✅ (2026-03-21, compare_models.py, test split)
- [x] **4.4.7** Внесено в generate_plots.py: PHASE5_MODELS, PHASE5_LOAO, fig12_phase5_loao, fig13_phase5_fps_map ✅

### 4.4 — Расширение TSM на все 6 топ-моделей

- [x] **4.4.8** Реализован `mobilenet_v3_small_tsm` backbone в `src/models/backbone.py` ✅
  - `wrap_inverted_residual_with_tsm()` — все InvertedResidual блоки обёрнуты в TemporalShift
- [x] **4.4.9** Извлечены фичи для `mv3_tsm`: 200+68+68=336 видео, 0 ошибок ✅ (2026-03-22, повтор после фикса TSM bug — `build_tsm_mobilenet_v3_small` не оборачивал блоки)
- [x] **4.4.10** `eff_b0_bilstm_tsm` (`efficientnet_b0_tsm_bilstm`): HPO best=0.895 | train mAP=0.874 | LOAO: Ball=0.946 ✅ Clubs=0.947 ✅ Hoop=0.819 ✅ Ribbon=0.545 ❌ | Mean=0.814±0.164 ❌ Ribbon (2026-03-23)
- [ ] ~~**4.4.11**~~ `mv3_bilstm_tsm`: **ПРИОСТАНОВЛЕН** — фокус на eff_b0
- [x] **4.4.12** `eff_b0_tcn_tsm` (`efficientnet_b0_tsm_tcn`): HPO best=0.843 | train mAP=0.636 | LOAO: Ball=0.779 ✅ Clubs=0.677 ❌ Hoop=0.597 ❌ Ribbon=0.282 ❌ | Mean=0.584±0.186 ❌ (2026-03-23)
- [ ] ~~**4.4.13**~~ `mv3_bilstm_attn_tsm`: **ПРИОСТАНОВЛЕН**
- [ ] ~~**4.4.14**~~ `mv3_tcn_tsm`: **ПРИОСТАНОВЛЕН**

---

## Задача 4.5: Архитектура D — Pose-based BiLSTM

**Цель:** Полностью apparatus-agnostic подход через скелет гимнастки.

- [x] **4.5.1** Реализован `src/scripts/extract_pose_features.py` ✅
  - MediaPipe Pose → 33 точки × (x, y, visibility) = 99 чисел/кадр
  - Нормализация: центр = midpoint бёдер (23, 24); масштаб = нос(0)–щиколотка(27)
  - Кэш: `data/pose_features/<split>/<stem>.pt` → `{features: [F, 99], fps: float}`
- [x] **4.5.2** Реализован `src/data/pose_dataset.py` (аналог `frame_dataset.py`, интерфейс идентичен) ✅
- [x] **4.5.3** Реализован `src/models/pose_model.py` ✅
  - `PoseHead`: LayerNorm + Linear(99→hidden_dim) + ReLU → BiLSTM+Attention
  - `PoseGymRT`: обёртка с `.temporal`, `.size_mb()`, `.count_parameters()` — совместим с train.py/hpo.py/loao_cv.py
  - Параметры: 1,392,712 | 5.31 MB (hidden_dim=256, n_layers=2)
- [x] **4.5.4** Создан `configs/pose_bilstm_attn.yaml` (base конфиг для HPO) ✅
  - train.py / hpo.py / loao_cv.py патчены: поддержка `model_type: "pose"`
- [x] **4.5.4b** Извлечение pose-фич завершено: train=200 ✅ valid=68 ✅ test=68 ✅ (2026-03-22, VIDEO mode, Tasks API)
- [x] **4.5.5** HPO 20 trials: best=1.000, best params: lr=4.3e-4, hidden_dim=64, n_layers=3, chunk_size=358 ✅ (2026-03-23)
- [x] **4.5.6** Обучить seed=42: mAP@0.5=0.856, BE=0.227s, Precision=0.936, Recall=0.921 ✅ (2026-03-23)
- [x] **4.5.7** LOAO: Ball=0.989 ✅ Clubs=0.655 ❌ Hoop=1.000 ✅ Ribbon=1.000 ✅ | Mean=0.911±0.148 ❌ Clubs (2026-03-23) — **Ribbon впервые 1.000!**
- [x] **4.5.8** FPS измерен (compare_models.py, 2026-03-24): pose_causal_tcn=1,250,731 FPS, pose_bilstm_attn=310,922 FPS, pose_bilstm=17,852 FPS (pre-extracted features; реальный bottleneck = MediaPipe) ✅
- [x] **4.5.9** Внести в графики — pose модели в fig14 scatter, fig15 LOAO heatmap, fig16 LOAO bar ✅ (2026-03-24)
- [x] **4.5.10a** `pose_bilstm` HPO: best=0.994 (trial 4) | lr=1.29e-4, hidden_dim=256, n_layers=2, dropout=0.486, chunk_size=439 ✅ (2026-03-23)
- [x] **4.5.10b** `pose_bilstm` train: mAP@0.5=0.994, BE=0.19s, Recall=0.968 → `pose_bilstm_opt_seed42_best.pt` ✅ (2026-03-23)
- [x] **4.5.10c** `pose_bilstm` LOAO: Ball=0.989 ✅ Clubs=0.903 ✅ Hoop=0.994 ✅ Ribbon=1.000 ✅ | Mean=0.972±0.040 **PASS** ✅ (2026-03-24)
- [x] **4.5.11a** `pose_causal_tcn` HPO: best=0.997 (trial 10) | lr=4.03e-3, dropout=0.247, chunk_size=374 ✅ (2026-03-23)
- [x] **4.5.11b** `pose_causal_tcn` train: mAP@0.5=0.997, BE=0.45s, Recall=0.889 → `pose_causal_tcn_opt_seed42_best.pt` ✅ (2026-03-23)
- [x] **4.5.11c** `pose_causal_tcn` LOAO: Ball=0.973 ✅ Clubs=0.989 ✅ Hoop=1.000 ✅ Ribbon=0.909 ✅ | Mean=0.968±0.035 **PASS** ✅ (2026-03-24)

---

## Задача 4.6: Финальные графики и выводы

- [x] **4.6.1** Scatter plot **Accuracy vs Speed** — все модели Phase 3 + framediff + TSM + Pose + S3D ✅ (2026-03-24)
  (`data/plots/results/fig14_p5_accuracy_vs_speed.png`)
- [x] **4.6.2** **LOAO heatmap** полный — 17 моделей × 4 снаряда + Mean ✅ (2026-03-24)
  (`data/plots/results/fig15_p5_loao_heatmap_full.png`)
- [x] **4.6.3** **LOAO mean bar chart** — все архитектуры, threshold 0.70, pose [PASS] выделены ✅ (2026-03-24)
  (`data/plots/results/fig16_p5_loao_mean_bar.png`)
- [x] **4.6.4** Обновить `docs/thesis_research_notes.md` с финальными выводами ✅ (2026-03-24):
  - [x] «2D backbone = apparatus shortcuts, LOAO ❌» — задокументировано
  - [x] «Framediff: улучшает Ball/Clubs/Hoop, Ribbon провален (0.429); mean 0.783» — задокументировано
  - [x] «TSM: сдвигает shortcuts — Hoop +0.267, Ribbon −0.179; mean 0.829» — задокументировано
  - [x] «TSM > framediff по Mean и Ribbon» — задокументировано
  - [x] «S3D: FPS=261.9 (CNN+decode), но latency=640ms → нарушает ≤40ms; LOAO=0.879 ✅» — задокументировано
  - [x] «Pose: LOAO ✅ для pose_bilstm (0.972) и pose_causal_tcn (0.968) — первые RT-модели с PASS» — задокументировано
  - [x] test mAP таблица всех Phase 4 моделей — добавлена
- [x] **4.6.5** Обновлён `docs/comparison_table.md` (compare_models.py, 2026-03-24) ✅

---

---

## Условие перехода к Фазе 5

**Обновлено 2026-03-24:** pose_bilstm и pose_causal_tcn ПРОШЛИ LOAO (все 4 снаряда ≥ 0.70).

**Итоговый вывод Phase 4:**
- CNN + framediff/TSM: улучшают LOAO Mean, но не достигают критерия (Ribbon < 0.70 у всех)
- Pose-based: **первые RT-модели, прошедшие LOAO** — apparatus-agnostic признаки решают проблему shortcuts
  - pose_bilstm: LOAO=0.972 ✅, test mAP=0.875
  - pose_causal_tcn: LOAO=0.968 ✅, test mAP=0.982 (лучший Phase 4)
  - pose_bilstm_attn: LOAO=0.911 ❌ (Clubs=0.655)
- S3D: LOAO=0.879 ✅, FPS=261.9, но latency 640ms нарушает ≤40ms RT-требование

**Гипотеза Phase 5 (пока не планируется):** двухпоточная архитектура CNN(TSM) + Pose → конкатенация → LSTM как потенциальное улучшение.

---

## Критерии завершения Фазы 4

*(Скоуп сужен до eff_b0 — mv3 приостановлен, S3D отложен)*

- [x] framediff протестирован на eff_b0 (bilstm_attn ✅, bilstm ✅, tcn ✅)
- [x] TSM протестирован на eff_b0 (bilstm_attn ✅, bilstm ✅, tcn ✅)
- [x] Pose-based (задача 4.5): pose_bilstm_attn ✅, pose_bilstm ✅, pose_causal_tcn ✅
- [x] **4.5.8** FPS измерен (compare_models.py, 2026-03-24) ✅
- [x] **4.6** Финальные графики: fig14 scatter ✅, fig15 LOAO heatmap ✅, fig16 LOAO mean bar ✅ (2026-03-24)
- [x] **4.6.4** thesis_research_notes.md — финальные выводы Phase 4 ✅ (2026-03-24)
- [x] `pytest tests/` — 246 passed, 0 errors ✅ (2026-03-24)
- [x] Коммит: "Phase 4: generalization ablation — framediff, TSM, pose (eff_b0)" ✅ (2026-03-24, c36abb7)
