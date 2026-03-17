"""
StreamingDetector — онлайн детектор элемента для реального времени.

Принимает покадровые scores от temporal head и выдаёт состояние детекции.

Алгоритм:
  - EMA сглаживание: ema = alpha * score + (1 - alpha) * ema_prev
  - Гистерезис: активируется при ema > threshold_high,
    деактивируется если ema < threshold_low на протяжении patience_frames подряд
  - Duration filter: сегменты короче min_frames или длиннее max_frames игнорируются

Использование:
  detector = StreamingDetector(cfg["streaming"])
  for frame_idx, score in enumerate(scores):
      state = detector.update(score, frame_idx)
      if state.is_active:
          print(f"Element at frame {state.start_frame}...")
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionState:
    """Текущее состояние детектора после обработки одного кадра."""
    is_active: bool          # идёт ли элемент прямо сейчас
    start_frame: int | None  # кадр начала текущего/последнего сегмента (None если не было)
    current_score: float     # сырой score текущего кадра
    ema_score: float         # EMA-сглаженный score


@dataclass
class _CompletedSegment:
    start_frame: int
    end_frame: int


class StreamingDetector:
    """Онлайн детектор с гистерезисом и EMA.

    Args:
        threshold_high:  порог активации (ema > high → старт сегмента)
        threshold_low:   порог деактивации (ema < low N кадров подряд → конец сегмента)
        patience_frames: сколько кадров ema может быть ниже threshold_low до деактивации
        ema_alpha:       вес нового score в EMA (0 < alpha ≤ 1)
        min_frames:      минимальная длина сегмента в кадрах (0 = без ограничения)
        max_frames:      максимальная длина сегмента в кадрах (0 = без ограничения)
    """

    def __init__(
        self,
        threshold_high: float = 0.6,
        threshold_low: float = 0.3,
        patience_frames: int = 15,
        ema_alpha: float = 0.3,
        min_frames: int = 0,
        max_frames: int = 0,
    ) -> None:
        if not (0 < ema_alpha <= 1):
            raise ValueError(f"ema_alpha must be in (0, 1], got {ema_alpha}")
        if threshold_low > threshold_high:
            raise ValueError(
                f"threshold_low ({threshold_low}) must be <= threshold_high ({threshold_high})"
            )

        self.threshold_high = threshold_high
        self.threshold_low = threshold_low
        self.patience_frames = patience_frames
        self.ema_alpha = ema_alpha
        self.min_frames = min_frames
        self.max_frames = max_frames

        self._ema: float = 0.0
        self._is_active: bool = False
        self._start_frame: int | None = None
        self._low_counter: int = 0          # кадры подряд ниже threshold_low
        self.completed: list[_CompletedSegment] = field(default_factory=list)
        self.completed = []

    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Сбрасывает внутреннее состояние (новое видео)."""
        self._ema = 0.0
        self._is_active = False
        self._start_frame = None
        self._low_counter = 0
        self.completed = []

    def update(self, score: float, frame_idx: int) -> DetectionState:
        """Обрабатывает один кадр.

        Args:
            score:     сырой score от temporal head (после sigmoid), ∈ [0, 1]
            frame_idx: индекс кадра (0-based)

        Returns:
            DetectionState с текущим состоянием детектора
        """
        # EMA
        self._ema = self.ema_alpha * score + (1.0 - self.ema_alpha) * self._ema

        if not self._is_active:
            # Ждём активации
            if self._ema > self.threshold_high:
                self._is_active = True
                self._start_frame = frame_idx
                self._low_counter = 0
        else:
            # Активны — проверяем деактивацию
            if self._ema < self.threshold_low:
                self._low_counter += 1
            else:
                self._low_counter = 0

            # Превышение max_frames → принудительно деактивируем
            active_len = frame_idx - self._start_frame + 1
            force_end = self.max_frames > 0 and active_len >= self.max_frames

            if self._low_counter >= self.patience_frames or force_end:
                end_frame = frame_idx
                seg_len = end_frame - self._start_frame + 1
                # Duration filter
                long_enough = self.min_frames == 0 or seg_len >= self.min_frames
                if long_enough:
                    self.completed.append(
                        _CompletedSegment(self._start_frame, end_frame)
                    )
                self._is_active = False
                self._start_frame = None
                self._low_counter = 0

        return DetectionState(
            is_active=self._is_active,
            start_frame=self._start_frame,
            current_score=score,
            ema_score=self._ema,
        )

    def finalize(self, last_frame_idx: int) -> DetectionState:
        """Принудительно завершает активный сегмент (конец видео).

        Вызывать после последнего кадра если нужно закрыть открытый сегмент.
        """
        if self._is_active and self._start_frame is not None:
            seg_len = last_frame_idx - self._start_frame + 1
            long_enough = self.min_frames == 0 or seg_len >= self.min_frames
            if long_enough:
                self.completed.append(
                    _CompletedSegment(self._start_frame, last_frame_idx)
                )
            self._is_active = False
            self._start_frame = None
            self._low_counter = 0

        return DetectionState(
            is_active=False,
            start_frame=None,
            current_score=0.0,
            ema_score=self._ema,
        )

    @classmethod
    def from_config(cls, cfg: dict) -> "StreamingDetector":
        """Создаёт детектор из секции 'streaming' конфига."""
        fps = cfg.get("fps", 25.0)
        min_dur = cfg.get("min_duration_sec", 1.5)
        max_dur = cfg.get("max_duration_sec", 12.0)
        return cls(
            threshold_high=cfg.get("threshold_high", 0.6),
            threshold_low=cfg.get("threshold_low", 0.3),
            patience_frames=cfg.get("patience_frames", 15),
            ema_alpha=cfg.get("ema_alpha", 0.3),
            min_frames=int(min_dur * fps) if min_dur > 0 else 0,
            max_frames=int(max_dur * fps) if max_dur > 0 else 0,
        )
