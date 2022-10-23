import math
import textwrap
import warnings
from typing import Union

from composer.core import State, Time, TimeUnit
from composer.optim.scheduler import ComposerScheduler


def _convert_time(time: Union[str, Time[int], Time[float]], state: State, ssr: float = 1.0) -> Time[int]:
    if isinstance(time, str):
        time = Time.from_timestring(time)

    assert state.max_duration is not None, 'max_duration should be set whenever schedulers are invoked'

    if time.unit == TimeUnit.DURATION:
        if state.max_duration.unit == TimeUnit.EPOCH:
            if state.dataloader_len is None:
                raise RuntimeError('Cannot convert time, as state.dataloader_len is None.')
            return Time(int(time.value * int(state.dataloader_len) * state.max_duration.value), TimeUnit.BATCH)
        return Time(int(time.value * state.max_duration.value), state.max_duration.unit)
    elif time.unit == TimeUnit.EPOCH:
        # Epochs do not provide sufficient granularity for SSR scaling
        # e.g. if max_duration = 1ep, then any SSR would result in a new duration of 0.
        # so, convert the time into batches
        if state.dataloader_len is None:
            raise RuntimeError('Cannot convert time, as state.dataloader_len is None.')
        time = Time(value=time.value * int(state.dataloader_len), unit=TimeUnit.BATCH)

    return Time(value=int(time.value * ssr), unit=time.unit)


def _cosine_anneal(x: float, min_y: float = 0.0, max_y: float = 1.0) -> float:
    """Implements a cosine decay curve.

    Curve is cos(x) on domain [0, pi], stretched to the domain [0, 1] and range [min_y, max_y]. Additionally, param x is
    clipped to the interval [0, 1]
    """
    x = min(max(x, 0.0), 1.0)
    return min_y + (max_y - min_y) * (1 + math.cos(x * math.pi)) / 2


class LinearScheduler(ComposerScheduler):
    def __init__(self, alpha_i: float = 1.0, alpha_f: float = 0.0, t_max: Union[str, Time] = '1dur'):
        self.alpha_i = alpha_i
        self.alpha_f = alpha_f
        self.t_max = t_max

    def __call__(self, state: State, ssr: float = 1.0):
        if isinstance(self.t_max, str):
            self.t_max = _convert_time(self.t_max, state, ssr=ssr)
        current_time = state.timestamp.get(self.t_max.unit)
        frac_of_total = min(1.0, (current_time / self.t_max).value)

        current_factor = self.alpha_i + frac_of_total * (self.alpha_f - self.alpha_i)

        return current_factor


class CosineAnnealingWithWarmupScheduler(ComposerScheduler):
    def __init__(self,
                 t_warmup: Union[str, Time],
                 surgery_time: Time,
                 bs_multiplier: int,
                 t_max: Union[str, Time] = '1dur',
                 alpha_f: float = 0.0,
                 scale_warmup: bool = False,
    ):
        self.t_warmup = t_warmup
        self.surgery_time = surgery_time
        if isinstance(surgery_time, str):
            self.surgery_time = Time.from_timestring(surgery_time)
            assert self.surgery_time.unit == TimeUnit.EPOCH
        self.bs_multiplier = bs_multiplier
        self.t_max = None  # will be set during 1st call to scheduler
        self.alpha_f = alpha_f
        self.scale_warmup = scale_warmup
        self.warmup_scheduler = LinearScheduler(alpha_i=0.0, alpha_f=1.0, t_max=t_warmup)

    def __call__(self, state: State, ssr: float = 1.0):

        # set them once during the 1st call
        if isinstance(self.t_warmup, str):
            self.t_warmup = Time.from_timestring(self.t_warmup)
            assert self.t_warmup.unit == TimeUnit.EPOCH
            self.t_warmup = _convert_time(self.t_warmup, state)
            assert self.t_warmup.unit == TimeUnit.BATCH

        if self.t_max is None:
            assert state.max_duration.unit == TimeUnit.EPOCH
            assert self.surgery_time.unit == TimeUnit.EPOCH
            assert state.dataloader_len.unit == TimeUnit.BATCH
            pre_surgery_duration = self.surgery_time.value * int(state.dataloader_len)
            post_surgery_duration = (int(state.max_duration.value * ssr) - self.surgery_time.value) * (state.dataloader_len.value // self.bs_multiplier)
            self.t_max = Time(pre_surgery_duration + post_surgery_duration, TimeUnit.BATCH)

        # rest same as original code
        if self.t_warmup.value == 0:
            warnings.warn(
                textwrap.dedent("""\
                The warmup duration is 0. If you specified warmup as a fraction of total
                training duration, take note that the warmup duration is calculated in the
                same unit as the trainer's max_duration parameter."""))

        if state.timestamp < self.t_warmup:
            if self.scale_warmup:
                return self.warmup_scheduler(state, ssr)
            return self.warmup_scheduler(state)

        current_time = state.timestamp.get(self.t_warmup.unit)
        frac_of_total = ((current_time - self.t_warmup) / (self.t_max - self.t_warmup)).value if (self.t_max > self.t_warmup) else 0.0
        frac_of_total = min(1.0, frac_of_total)
        return _cosine_anneal(x=frac_of_total, min_y=self.alpha_f)
