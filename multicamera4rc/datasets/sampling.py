"""Dataset quotas and constant-memory permutations of native coverage windows."""

import math
import random


def source_quotas(weights, epoch_size):
    """Largest-remainder apportionment; pool size never changes source ratios."""
    total = sum(weights)
    exact = [epoch_size * weight / total for weight in weights]
    quotas = [math.floor(value) for value in exact]
    order = sorted(range(len(weights)), key=lambda i: (exact[i] - quotas[i], -i), reverse=True)
    for i in order[:epoch_size - sum(quotas)]:
        quotas[i] += 1
    if not all(quotas):
        raise ValueError("balanced_coverage epoch_size is too small to sample every source")
    return quotas


def coverage_permutation(size, seed):
    """An affine bijection, so successive visits eventually traverse every window."""
    rng = random.Random(seed)
    multiplier = rng.randrange(1, max(2, size))
    while math.gcd(multiplier, size) != 1:
        multiplier = multiplier % max(1, size - 1) + 1
    return multiplier, rng.randrange(size)
