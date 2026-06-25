import dask_cudf
from features.gpu_vector_builder import build_gpu_vector
from utils.logger import get_logger

logger = get_logger("multi_gpu_vector_builder")

def build_multi_gpu_vector(dask_df):
    logger.info(f"Building multi-GPU vectors for {len(dask_df)} rows...")
    return dask_df.map_partitions(build_gpu_vector).persist()
