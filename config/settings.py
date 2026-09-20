from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{ROOT / 'data' / 'breast_insight.db'}")
DATASET_PATH = ROOT / "data" / "data.csv"
RANDOM_STATE = 42
BOOTSTRAP_ITERATIONS = 5000
