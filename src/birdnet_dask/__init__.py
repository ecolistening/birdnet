import pathlib
import pandas as pd
import logging

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
    "embeddings",
    "species_probs",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

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
        metadata.file_path,
        lat=metadata.get("latitude"),
        lon=metadata.get("longitude"),
        date=ts.date() if pd.notnull(ts := metadata.get("timestamp")) else None,
        min_conf=min_conf,
        **kwargs,
    )

    recording.analyze()

    df = pd.DataFrame(recording.detections)
    df["path"] = metadata.file_path
    df["file"] = pathlib.Path(metadata.file_path).name
    df["min_conf"] = min_conf
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df

@suppress_output()
def _embed_as_df(
    analyzer: Analyzer,
    metadata: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(metadata.file_path),
        lat=metadata.get("latitude"),
        lon=metadata.get("longitude"),
        date=ts.date() if pd.notnull(ts := metadata.get("timestamp")) else None,
        **kwargs,
    )

    recording.extract_embeddings()

    df = pd.DataFrame([
        pd.concat([
            pd.Series({ str(dim): value for dim, value in enumerate(embedding_info["embeddings"]) }),
            pd.Series({ k: v for k, v in embedding_info.items() if k != "embeddings" }),
        ])
        for embedding_info in recording.embeddings
    ])
    df["path"] = str(metadata.file_path)
    df["file"] = pathlib.Path(metadata.file_path).name
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df

