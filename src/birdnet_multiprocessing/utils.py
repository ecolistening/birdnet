import contextlib
import pandas as pd
import os
import sys
from typing import Any, List, Callable

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items.iloc[i:i + batch_size]

def try_or(func: Callable, default: Any) -> Any:
    try:
        return func()
    except Exception as e:
        return default

def read_metadata_file(path, **kwargs):
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
