import os
import logging
import pandas as pd
import soundfile

from tqdm import tqdm
from typing import Any

from birdnet_multiproceessing.multiproccessing import run_processing

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
    assert "file_path" in columns, "'file_path' column must be specified"
    assert "file_id" in columns, "'file_id' column specifying a unique ID for each file must be specified"

def valid_audio_file(file_path):
    try:
        soundfile.info(file_path)
        yield file_path, True
    except soundfile.LibsndfileError as e:
        log.warning(e)
        yield file_path, False

def validate_all_audio(file_paths, num_workers: int = 0):
    sync = num_workers == 0

    valid_file_paths = []
    invalid_file_paths = []

    with tqdm(total=len(file_paths)) as pbar:
        pbar.set_description("Validating audio before processing...")
        for file_path, valid in run_processing(valid_audio_file, inputs, num_workers=num_workers):
            if valid:
                valid_file_paths.append(file_path)
            elif invalid:
                invalid_file_paths.append(file_path)
            pbar.update(1)

    return valid_file_paths, invalid_file_paths

def valid_data(audio_dir, df):
    if (audio_dir / "failed_files.parquet").exists():
        invalid = pd.read_parquet(audio_dir / "failed_files.parquet")
        return df[~df.file_id.isin(invalid.file_id)]

    valid, invalid = validate_all_audio(df["file_path"])

    if len(invalid_df := df[df.file_path.isin(invalid)]):
        invalid_df.to_parquet(audio_dir / "failed_files.parquet")
        log.warning(f"Failed file references saved in '{audio_dir / 'failed_files.parquet'}'")

    return df[df.file_path.isin(valid)]
