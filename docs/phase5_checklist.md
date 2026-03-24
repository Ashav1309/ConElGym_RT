# Фаза 5: MobileNetV4 Backbone

**Цель:** Проверить, улучшает ли современный backbone (MobileNetV4, Google 2024) обобщаемость по снарядам (LOAO) по сравнению с EfficientNet-B0.

**Предусловие:** Фаза 4 завершена, лучшая модель Phase 4 выбрана как baseline.

---

## Мотивация

MobileNetV4 (Google, 2024) — специально спроектирован для edge/RT задач. Превосходит EfficientNet-B0 по ImageNet при сопоставимой скорости.

| Backbone | Параметры | ImageNet top-1 | Год | Библиотека |
|----------|:---------:|:--------------:|:---:|:----------:|
| EfficientNet-B0 | 5.3M | 77.1% | 2019 | torchvision |
| MobileNetV4-Conv-Small | 3.8M | 73.8% | 2024 | timm |
| MobileNetV4-Conv-Small | 3.8M | 73.8% | 2024 | timm |

**Вариант для эксперимента:** `mobilenetv4_conv_small` — лучший баланс accuracy/speed, превосходит eff_b0 по качеству признаков.

---

## Задача 5.1: Подготовка backbone

- [ ] **5.1.1** Установить `timm`: `pip install timm`; добавить в `requirements.txt`
- [ ] **5.1.2** Добавить `mobilenetv4_conv_small` backbone в `src/models/backbone.py`:
  - `timm.create_model('mobilenetv4_conv_small.e500_r256_in1k', pretrained=True, num_classes=0)`
  - `output_dim=1280` (после global avg pool)
- [ ] **5.1.3** Извлечь фичи: `python src/scripts/extract_frame_features.py --backbone mobilenetv4_conv_small --split all`
  - train=200 ✅, valid=68 ✅, test=68 ✅

---

## Задача 5.2: Обучение

- [ ] **5.2.1** Создать `configs/mv4_small_bilstm_attn_opt.yaml`
  - Базовая архитектура: BiLSTM+Attention (лучшая из Phase 3)
  - `backbone: mobilenetv4_conv_small`, `backbone_dim: 960`
- [ ] **5.2.2** HPO 20 trials (пространство поиска идентично Phase 3)
- [ ] **5.2.3** Обучить seed=42: зафиксировать mAP@{0.3,0.5,0.7}, Precision, Recall, BE

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

| Метрика | eff_b0_bilstm_attn | mv4_medium_bilstm_attn | Δ |
|---------|--------------------|------------------------|---|
| mAP@0.5 (valid) | 0.856 | | |
| mAP@0.5 (test) | 0.788 | | |
| LOAO Mean | 0.739 | | |
| LOAO Ribbon | 0.800 | | |
| FPS (e2e) | 358 | | |
| Size MB | 26.5 | | |

---

## Критерии завершения Фазы 5

- [ ] `mobilenetv4_conv_small` backbone добавлен и проверен
- [ ] Фичи извлечены для всех 3 сплитов (336 видео)
- [ ] HPO завершён (20 trials)
- [ ] seed=42 обучен, метрики зафиксированы
- [ ] LOAO завершён (все 4 снаряда)
- [ ] FPS измерен (e2e)
- [ ] Результаты добавлены в графики и `thesis_research_notes.md`
- [ ] `pytest tests/` — 0 ошибок
- [ ] Коммит: "Phase 5: MobileNetV4 backbone evaluation"
