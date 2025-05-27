import pytest
import warnings

import pandas as pd
import pathlib

from typing import Iterable

from birdnet_multiprocessing.main import (
    species_probs,
    species_probs_multiprocessing,
)
from birdnet_multiprocessing.utils import chunked

@pytest.fixture
def audio_dir(request):
    return pathlib.Path(request.config.getoption("--audio-dir"))

@pytest.fixture
def audio_path(request):
    return pathlib.Path(__file__).parent / "fixtures"  / "PL-12_0_20150603_0430.wav"

def test_process_audio_file(audio_path):
    df = species_probs(audio_path)
    assert isinstance(df, pd.DataFrame)
    for col in ["common_name", "scientific_name", "start_time", "end_time", "confidence", "label", "file_path"]:
        assert col in df.columns
    assert df.iloc[0]["file_path"] == audio_path

def test_process_audio_file_with_metadata(audio_path):
    df = species_probs(audio_path, latitude=0.0, longitude=0.0)
    assert isinstance(df, pd.DataFrame)
    for col in ["common_name", "scientific_name", "start_time", "end_time", "confidence", "label", "file_path", "latitude", "longitude"]:
        assert col in df.columns
    assert df.iloc[0]["file_path"] == audio_path

def test_process_audio_files_mp(audio_dir):
    if audio_dir.exists():
        inputs = [dict(file_path=file_path) for file_path in pathlib.Path(audio_dir).rglob('*.wav')]
        num_inputs = min(len(inputs), 10)
        pending = species_probs_multiprocessing(inputs[:num_inputs], num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))

def test_process_audio_files_with_metadata_mp(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = pd.read_parquet(audio_dir / "metadata.parquet")
        metadata["file_path"] = metadata["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
        metadata["date"] = metadata["timestamp"].dt.date
        inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")
        num_inputs = min(len(inputs), 10)
        pending = species_probs_multiprocessing(inputs[:num_inputs], num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))

def test_batch_process_audio_files_with_metadata_mp(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = pd.read_parquet(audio_dir / "metadata.parquet")
        metadata["file_path"] = metadata["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
        metadata["date"] = metadata["timestamp"].dt.date
        inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")
        num_inputs = min(len(inputs), 10)
        batches = list(chunked(inputs[:num_inputs], 6))
        pending = species_probs_multiprocessing(batches, num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))
