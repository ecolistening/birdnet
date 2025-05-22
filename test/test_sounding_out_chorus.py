import pytest

import pandas as pd
import pathlib

from birdnet_multiprocessing.main import (
    process_audio_file_with_birdnet,
    process_audio_files_with_birdnet_mp,
)
from birdnet_multiprocessing.utils import chunked

@pytest.fixture
def audio_dir(request):
    return pathlib.Path(request.config.getoption("--audio-dir"))

def test_process_audio_files_with_birdnet_mp(audio_dir):
    metadata = pd.read_parquet(audio_dir / "metadata.parquet")
    metadata["file_path"] = metadata["file_name"].map(lambda file_name: audio_dir / "data" / file_name)
    metadata["date"] = metadata["timestamp"].dt.date
    inputs = metadata[["file_path", "latitude", "longitude", "date"]].to_dict(orient="records")[:10]
    process_audio_files_with_birdnet_mp(list(chunked(inputs, 6)), num_workers=4)
