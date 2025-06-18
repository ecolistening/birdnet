import os
import logging
import pandas as pd
import soundfile

from tqdm import tqdm
from typing import Any

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def load_metadata_file(path: str, **kwargs: Any):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        return pd.read_csv(path, **kwargs)
    elif ext == '.parquet':
        return pd.read_parquet(path, **kwargs)
    elif ext in ['.xls', '.xlsx']:
        return pd.read_excel(path, **kwargs)
    elif ext == '.json':
        return pd.read_json(path, **kwargs)
    elif ext == '.feather':
        return pd.read_feather(path, **kwargs)
    elif ext == '.orc':
        return pd.read_orc(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def save_metadata_file(df: pd.DataFrame, path: str, **kwargs: Any):
    ext = os.path.splitext(path)[1].lower()

    if ext == '.csv':
        df.to_csv(path, index=False, **kwargs)
    elif ext == '.parquet':
        df.to_parquet(path, index=False, **kwargs)
    elif ext in ['.xls', '.xlsx']:
        df.to_excel(path, index=False, **kwargs)
    elif ext == '.json':
        df.to_json(path, **kwargs)
    elif ext == '.feather':
        df.to_feather(path, **kwargs)
    elif ext == '.orc':
        df.to_orc(path, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")

def validate_columns(columns):
    assert "file" in columns, "'file' column must be specified"
    assert "path" in columns, "'path' column must be specified"

