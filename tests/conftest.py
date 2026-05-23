import os
import sys
from pathlib import Path


os.environ.setdefault("MODEL_URI", "")
os.environ.setdefault("LOCAL_MODEL_PATH", "ml/artifacts/spam_model.joblib")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
