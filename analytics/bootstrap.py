import numpy as np

def bootstrap_mean_difference(group_a, group_b, iterations=5000, seed=42):
    rng = np.random.default_rng(seed)
    a = np.asarray(group_a, dtype=float)
    b = np.asarray(group_b, dtype=float)

    observed = a.mean() - b.mean()
    distribution = np.empty(iterations)

    for i in range(iterations):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        distribution[i] = sample_a.mean() - sample_b.mean()

    low, high = np.percentile(distribution, [2.5, 97.5])

    return {
        "observed": observed,
        "distribution": distribution,
        "ci_low": low,
        "ci_high": high
    }
