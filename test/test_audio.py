import datetime as dt
import itertools
import pandas as pd
import pathlib
import pytest
import numpy as np
import random
import warnings
import pyarrow.dataset as ds

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
        inputs = pd.DataFrame([
            dict(file_path=file_path)
            for file_path in itertools.islice(pathlib.Path(audio_dir).rglob('*.wav'), 10)
        ])
        pending = species_probs_multiprocessing(inputs, num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir)} does not exist, test failing quietly"))

def test_species_probs_from_audio_with_metadata_multiprocessing(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = ds.dataset(audio_dir / "metadata.parquet").scanner().head(10).to_pandas()
        metadata["file_path"] = metadata["file_path"].map(lambda file_path: str(audio_dir / file_path))
        metadata["timestamp"] = metadata.apply(lambda x: random_datetime() if np.random.rand() > 0.1 else np.nan, axis=1)
        metadata["latitude"] = np.where(np.random.rand(len(metadata)) < 0.1, np.nan, np.random.uniform(-90, 90, len(metadata)))
        metadata["longitude"] = np.where(np.random.rand(len(metadata)) < 0.1, np.nan, np.random.uniform(-180, 180, len(metadata)))
        inputs = metadata[["file_path", "latitude", "longitude", "timestamp"]]
        pending = species_probs_multiprocessing(inputs, num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir / 'metadata.parquet')} does not exist, test failing quietly"))

def test_batch_species_probs_from_audio_with_metadata_multiprocessing(audio_dir):
    if (audio_dir / "metadata.parquet").exists():
        metadata = ds.dataset(audio_dir / "metadata.parquet").scanner().head(10).to_pandas()
        metadata["file_path"] = metadata["file_path"].map(lambda file_path: str(audio_dir / file_path))
        metadata["timestamp"] = metadata.apply(lambda x: random_datetime() if np.random.rand() > 0.1 else np.nan, axis=1)
        metadata["latitude"] = np.where(np.random.rand(len(metadata)) < 0.1, np.nan, np.random.uniform(-90, 90, len(metadata)))
        metadata["longitude"] = np.where(np.random.rand(len(metadata)) < 0.1, np.nan, np.random.uniform(-180, 180, len(metadata)))
        inputs = metadata[["file_path", "latitude", "longitude", "timestamp"]]
        pending = species_probs_multiprocessing(inputs, num_workers=4, batch_size=6)
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
        inputs = pd.DataFrame([
            dict(file_path=file_path)
            for file_path in itertools.islice(pathlib.Path(audio_dir).rglob('*.wav'), 10)
        ])
        pending = embeddings_multiprocessing(inputs, num_workers=4)
        assert isinstance(pending, Iterable)
        results = list(pending)
        assert all(type(x) == pd.DataFrame for x in results)
    else:
        warnings.warn(UserWarning(f"{str(audio_dir)} does not exist, test failing quietly"))
