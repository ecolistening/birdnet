import pytest

import pandas as pd
import pathlib

from birdnet_multiprocessing.main import (
    species_presence_probs,
    species_presence_probs_multiprocessing,
)
from birdnet_multiprocessing.utils import chunked

@pytest.fixture
def audio_dir(request):
    return pathlib.Path(request.config.getoption("--audio-dir"))

@pytest.fixture
def audio_path(request):
    return pathlib.Path(__file__).parent / "fixtures" / "KN-05_0_20150509_0430.wav"

def test_process_audio_file(audio_path):
    result = species_presence_probs(audio_path)
    assert isinstance(result, tuple)
    assert result[0] == audio_path
    assert isinstance(result[1], dict)

def test_process_audio_files(audio_dir):
    inputs = [dict(file_path=file_path) for file_path in pathlib.Path(audio_dir).rglob('*.wav')]
    num_inputs = min(len(inputs), 10)
    results = list(species_presence_probs_multiprocessing(inputs[:num_inputs], num_workers=4))
    assert len(results) == num_inputs
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][1], dict)

def test_process_audio_files_with_metadata(audio_dir):
    metadata = pd.read_parquet(audio_dir / "metadata.parquet")
    metadata["file_path"] = metadata["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    metadata["date"] = metadata["timestamp"].dt.date
    inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")
    num_inputs = min(len(inputs), 10)
    results = list(species_presence_probs_multiprocessing(inputs[:num_inputs], num_workers=4))
    assert len(results) == num_inputs
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][1], dict)

def test_batch_process_audio_files_with_metadata(audio_dir):
    metadata = pd.read_parquet(audio_dir / "metadata.parquet")
    metadata["file_path"] = metadata["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    metadata["date"] = metadata["timestamp"].dt.date
    inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")
    num_inputs = min(len(inputs), 10)
    batches = list(chunked(inputs[:10], 6))
    results = list(species_presence_probs_multiprocessing(batches, num_workers=4))
    assert len(results) == num_inputs
    assert isinstance(results[0], tuple)
    assert isinstance(results[0][1], dict)
