import pathlib
import pandas as pd
import logging
import re
import soundfile

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from birdnet_dask.analyzer import _fetch_analyzer
from birdnet_dask.utils import suppress_output

__all__ = [
    "list_audio_files",
    "valid_audio_file",
    "embeddings",
    "species_probs",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)
BIRDNET_EMBEDDING_DIM = 1024

def list_audio_files(audio_dir):
    return [
        dict(path=str(file_path), file=file_path.name)
        for file_path in pathlib.Path(audio_dir).rglob('*')
        if AUDIO_FILE_REGEX.match(str(file_path))
    ]

def valid_audio_file(metadata: Dict[str, str]):
    try:
        soundfile.info(metadata["path"])
        return True
    except soundfile.LibsndfileError as e:
        log.warning(e)
        return False

def species_probs_meta():
    return pd.DataFrame({
        "path": pd.Series(dtype="object"),
        "file": pd.Series(dtype="object"),
        "min_conf": pd.Series(dtype="float64"),
        "model": pd.Series(dtype="object"),
        "common_name": pd.Series(dtype="object"),
        "scientific_name": pd.Series(dtype="object"),
        "label": pd.Series(dtype="object"),
        "start_time": pd.Series(dtype="float64"),
        "end_time": pd.Series(dtype="float64"),
        "confidence": pd.Series(dtype="float64"),
    })

def embed_meta():
    return pd.DataFrame({
        "path": pd.Series(dtype="object"),
        "file": pd.Series(dtype="object"),
        "model": pd.Series(dtype="object"),
        "start_time": pd.Series(dtype="float64"),
        "end_time": pd.Series(dtype="float64"),
        **{ dim: pd.Series(dtype="float64") for dim in map(str, range(BIRDNET_EMBEDDING_DIM)) },
    })

def species_probs(
    df: pd.DataFrame,
    min_conf: float,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = _fetch_analyzer()
    return pd.concat([
        _species_probs_as_df(analyzer, metadata, min_conf=min_conf, **kwargs)
        for i, metadata in df.iterrows()
    ], axis=0)

def embed(
    df: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = _fetch_analyzer()
    return pd.concat([
        _embed_as_df(analyzer, metadata, **kwargs)
        for i, metadata in df.iterrows()
    ], axis=0)

@suppress_output()
def _species_probs_as_df(
    analyzer: Analyzer,
    metadata: pd.Series,
    min_conf: float,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        metadata.path,
        lat=metadata.get("latitude"),
        lon=metadata.get("longitude"),
        date=ts.date() if pd.notnull(ts := metadata.get("timestamp")) else None,
        min_conf=min_conf,
        **kwargs,
    )

    recording.analyze()

    template_df = species_probs_meta()
    if not len(recording.detections):
        return template_df
    else:
        df = pd.DataFrame(recording.detections)
        df["path"] = metadata.path
        df["file"] = pathlib.Path(metadata.path).name
        df["min_conf"] = min_conf
        df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
        return df[template_df.columns]

@suppress_output()
def _embed_as_df(
    analyzer: Analyzer,
    metadata: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(metadata.path),
        lat=metadata.get("latitude"),
        lon=metadata.get("longitude"),
        date=ts.date() if pd.notnull(ts := metadata.get("timestamp")) else None,
        **kwargs,
    )

    recording.extract_embeddings()

    df = pd.DataFrame([
        pd.concat([
            pd.Series({ dim: value for dim, value in zip(map(str, range(len(embedding_info["embeddings"]))), embedding_info["embeddings"]) }),
            pd.Series({ k: v for k, v in embedding_info.items() if k != "embeddings" }),
        ])
        for embedding_info in recording.embeddings
    ])
    df["path"] = str(metadata.path)
    df["file"] = pathlib.Path(metadata.path).name
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df[embed_meta().columns]
