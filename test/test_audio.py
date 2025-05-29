import datetime as dt
import pandas as pd
import pathlib
import pytest
import numpy as np
import random
import warnings

from typing import Iterable

from birdnet_multiprocessing.main import (
    species_probs,
    species_probs_multiprocessing,
    embeddings,
    embeddings_multiprocessing,
)
from birdnet_multiprocessing.utils import chunked

def random_datetime(start_year=1970, end_year=2100):
    start = dt.datetime(start_year, 1, 1)
    end = dt.datetime(end_year, 12, 31)
    delta = end - start
    int_delta = int(delta.total_seconds())
    random_second = random.randint(0, int_delta)
    return start + dt.timedelta(seconds=random_second)

@pytest.fixture
def audio_dir(request):
    return pathlib.Path(request.config.getoption("--audio-dir"))

@pytest.fixture
def audio_path(request):
    return pathlib.Path(__file__).parent / "fixtures"  / "PL-12_0_20150603_0430.wav"

def test_species_probs_from_audio(audio_path):
    data = pd.Series(dict(file_path=audio_path))
    df = species_probs(data)
    assert isinstance(df, pd.DataFrame)
    expected_columns = list(sorted([
        "common_name", "scientific_name",
        "start_time", "end_time",
        "confidence", "label",
        "file_path", "model",
    ]))
    actual_columns = list(sorted(df.columns, key=str))
    assert expected_columns == actual_columns
    assert df.iloc[0]["file_path"] == str(data.file_path)

def test_species_probs_from_audio_with_metadata(audio_path):
    data = pd.Series(dict(file_path=audio_path, latitude=0.0, longitude=0.0))
    df = species_probs(data)
    assert isinstance(df, pd.DataFrame)
    expected_columns = list(sorted([
        "common_name", "scientific_name",
        "start_time", "end_time",
        "confidence", "label",
        "file_path", "model",
    ], key=str))
    actual_columns = list(sorted(df.columns, key=str))
    assert expected_columns == actual_columns
    assert len(df["file_path"].unique()) == 1
    assert df.iloc[0]["file_path"] == str(data.file_path)

def test_species_probs_from_audio_multiprocessing(audio_dir):
    if audio_dir.exists():
        inputs = pd.DataFrame([dict(file_path=file_path) for file_path in pathlib.Path(audio_dir).rglob('*.wav')])
        num_inputs = min(len(inputs), 10)
        pending = species_probs_multiprocessing(inputs.iloc[:num_inputs], num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir)} does not exist, test failing quietly"))

def test_species_probs_from_audio_with_metadata_multiprocessing(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = pd.read_parquet(audio_dir / "metadata.parquet")
        metadata["file_path"] = metadata["file_path"].map(lambda file_path: str(audio_dir / file_path))
        metadata["timestamp"] = metadata.apply(lambda x: random_datetime(), axis=1)
        metadata["latitude"] = np.random.uniform(low=-90, high=90, size=(len(metadata)))
        metadata["latitude"] = np.random.uniform(low=-180, high=180, size=(len(metadata)))
        inputs = metadata[metadata.columns.intersection(["file_path", "latitude", "longitude", "timestamp"])]
        num_inputs = min(len(inputs), 10)
        pending = species_probs_multiprocessing(inputs.iloc[:num_inputs], num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))

def test_batch_species_probs_from_audio_with_metadata_multiprocessing(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = pd.read_parquet(audio_dir / "metadata.parquet")
        metadata["file_path"] = metadata["file_path"].map(lambda file_path: str(audio_dir / file_path))
        metadata["timestamp"] = metadata.apply(lambda x: random_datetime(), axis=1)
        metadata["latitude"] = np.random.uniform(low=-90, high=90, size=(len(metadata)))
        metadata["latitude"] = np.random.uniform(low=-180, high=180, size=(len(metadata)))
        inputs = metadata[metadata.columns.intersection(["file_path", "latitude", "longitude", "timestamp"])]
        num_inputs = min(len(inputs), 10)
        pending = species_probs_multiprocessing(inputs.iloc[:num_inputs], num_workers=4, batch_size=6)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))

def test_embeddings_from_audio(audio_path):
    data = pd.Series(dict(file_path=audio_path))
    df = embeddings(data)
    assert isinstance(df, pd.DataFrame)
    expected_columns = list(sorted(["file_path", "start_time", "end_time", "model", *list(range(1024))], key=str))
    actual_columns = list(sorted(df.columns, key=str))
    assert expected_columns == actual_columns
    assert len(df) == 20
    assert len(df["file_path"].unique()) == 1
    assert df.iloc[0]["file_path"] == str(data.file_path)

def test_embeddings_from_audio_multiprocessing(audio_dir):
    if audio_dir.exists():
        inputs = pd.DataFrame([dict(file_path=file_path) for file_path in pathlib.Path(audio_dir).rglob('*.wav')])
        num_inputs = min(len(inputs), 10)
        pending = embeddings_multiprocessing(inputs.iloc[:num_inputs], num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir)} does not exist, test failing quietly"))
