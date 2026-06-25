import cudf
from utils.logger import get_logger

logger = get_logger("gpu_feature_store")

class GPUFeatureStore:
    def __init__(self):
        self.store = cudf.DataFrame()

    def upsert(self, df, key_col="entity_id"):
        if self.store.empty:
            self.store = df
        else:
            self.store = self.store.merge(df, on=key_col, how="outer")
        logger.info(f"Feature store size: {len(self.store)}")

    def get(self, entity_ids, key_col="entity_id"):
        return self.store[self.store[key_col].isin(entity_ids)]
