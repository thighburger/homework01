from functools import lru_cache

import joblib

try:
    from app.config import LOCAL_MODEL_PATH
except ModuleNotFoundError:
    from config import LOCAL_MODEL_PATH


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(LOCAL_MODEL_PATH)
