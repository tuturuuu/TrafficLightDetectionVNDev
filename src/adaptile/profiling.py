"""Synchronized wall-time stages; inference totals still use an enclosing timer."""

from contextlib import contextmanager
import time

import torch


def synchronize(device):
    if device is not None and torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def measure(timings, name, device=None):
    synchronize(device)
    start = time.perf_counter()
    yield
    synchronize(device)
    timings[name] = timings.get(name, 0.0) + (time.perf_counter() - start) * 1000
