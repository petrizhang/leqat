import numba as nb
import numpy as np
from numba import jit


@jit(nopython=True)
def _dp_pruning(k: int, r: float, K: np.ndarray, PC: np.ndarray):
    N = len(K)
    W = int(np.ceil(k - k * r))
    # Note: the index of K, PC, and q start with 0
    dp = np.zeros(shape=(N + 1, W + 1), dtype=nb.int64)
    # the i-th item
    for i in range(0, N):
        dp_idx = i + 1
        for w in range(1, W + 1):
            # weight limit = w

            if K[i] > w:
                dp[dp_idx, w] = dp[dp_idx - 1, w]
            else:
                v0 = dp[dp_idx - 1, w]
                v1 = PC[i] + dp[dp_idx - 1, w - K[i]]
                dp[dp_idx, w] = max(v0, v1)

    q = np.zeros(N, dtype=nb.int64)
    w = W
    res = dp[-1][-1]
    for i in range(N - 1, -1, -1):
        if res < 0:
            break
        dp_idx = i + 1
        if res == dp[dp_idx - 1][w]:
            q[i] = 0
        else:
            res = res - PC[i]
            w = w - K[i]
            q[i] = 1

    b = 1 - q
    return b


@jit(nopython=True)
def practical_dp_pruning(k: int, sigma: float, y: np.ndarray, PC: np.ndarray):
    if sigma == 1:
        return np.arange(0, len(y), dtype=nb.int64)
    # calculate the first non-zero digit of 1-sigma
    # e.g. 0.1   < x <= 1    -> 1
    # e.g. 0.01  < x <= 0.1  -> 2
    #      0.001 < x <= 0.01 -> 3
    alpha = 1+np.ceil(-np.log10(1 - sigma))
    factor = np.power(10, alpha)
    #factor = 100

    K = (k * y * factor).astype(nb.int64)
    k = int(k * factor)

    # Note: _dp_pruning only works for non-zero elements in K
    b = np.zeros(len(K), dtype=nb.int32)
    mask = K != 0
    non_zero_K = K[mask]

    non_zero_PC = PC[mask]
    non_zero_b = _dp_pruning(k, sigma, non_zero_K, non_zero_PC)
    b[mask] = non_zero_b
    return np.where(b == 1)[0]


@jit(nopython=True)
def greedy_pruning(k: int, tau: float, y: np.ndarray, PC: np.ndarray):
    # print("tau:", tau)
    # print("y:", y)
    # k and PC are defined for a unified interface and will be fully ignored
    sorted_y = np.sort(y)[::-1]
    sorted_idx = np.argsort(-y)
    cumsum = np.cumsum(sorted_y)
    i = np.where(cumsum >= tau)[0][0]
    return sorted_idx[: i + 1]


if __name__ == "__main__":
    k = 10
    r = 0.7
    K = np.array([0, 0, 0.6, 0.1, 0.3, 0, 0])
    PC = np.array([10, 10, 10, 60, 40, 10, 10])
    b = practical_dp_pruning(k, r, K, PC)
    print(b)

    # y = np.array([0.2, 0.3, 0.5, 0.0])
    # p = greedy_pruning(k, r, y, PC)
    # print(p)

