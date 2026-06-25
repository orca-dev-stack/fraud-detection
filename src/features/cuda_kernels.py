import cupy as cp
from numba import cuda
from utils.logger import get_logger

logger = get_logger("cuda_kernels")

@cuda.jit
def amount_kernel(amounts, out_log, out_ratio, user_avg):
    i = cuda.grid(1)
    if i < amounts.size:
        a = amounts[i]
        out_log[i] = cp.log1p(a)
        out_ratio[i] = a / max(user_avg[i], 1e-6)

def fast_amount_features(amounts, user_avg):
    n = amounts.size
    out_log = cp.zeros(n, dtype=cp.float32)
    out_ratio = cp.zeros(n, dtype=cp.float32)

    threads = 256
    blocks = (n + threads - 1) // threads
    amount_kernel[blocks, threads](amounts, out_log, out_ratio, user_avg)

    logger.info(f"CUDA amount features for {n} rows.")
    return out_log, out_ratio
