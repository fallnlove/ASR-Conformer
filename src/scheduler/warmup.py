from typing import List

import torch
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """
    The WarmupLR scheduler
    from https://arxiv.org/pdf/1706.03762

    Code from https://github.com/wenet-e2e/wenet/blob/2d0da71db4023174027b39d8716804d039b74a67/wenet/utils/scheduler.py#L26
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int = 3000,
        last_epoch: int = -1,
    ):
        self.warmup_steps = warmup_steps
        # __init__() must be invoked before setting field
        # because step() is also invoked in __init__()
        super().__init__(optimizer, last_epoch)

    def __repr__(self):
        return f"{self.__class__.__name__}(warmup_steps={self.warmup_steps})"

    def get_lr(self):
        step_num = self.last_epoch + 1
        warmup_steps = self.warmup_steps
        if not isinstance(warmup_steps, List):
            warmup_steps = [self.warmup_steps] * len(self.base_lrs)

        def initlr_fn(lr):
            return lr * step_num**-0.5

        def warmuplr_fn(lr, warmup_step):
            return (
                lr
                * warmup_step**0.5
                * min(step_num**-0.5, step_num * warmup_step**-1.5)
            )

        return [
            initlr_fn(lr) if warmup_steps[i] == 0 else warmuplr_fn(lr, warmup_steps[i])
            for (i, lr) in enumerate(self.base_lrs)
        ]

    def set_step(self, step: int):
        self.last_epoch = step
