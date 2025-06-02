import argparse
import functools
import multiprocessing as mp
import pandas as pd
import logging

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from tqdm import tqdm

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from birdnet_multiprocessing.multiprocessing import run_processing, process_batched, process_sequentially
from birdnet_multiprocessing.utils import chunked, suppress_output, try_or

__ALL__ = [
    "embeddings",
    "species_probs",
    "embeddings_and_species_probs",
    "embeddings_multiprocessing",
    "species_probs_multiprocessing",
    "embeddings_and_species_probs_multiprocessing",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ------------------------- Main API ------------------------------- #

def species_probs(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Extract species probabilities for a single file
    """
    analyzer = Analyzer()
    return _species_probs_as_df(analyzer, file_path, **kwargs)

def species_probs_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    """
    Extract species probabilities for all file paths specified in a dataframe
    """
    total = len(df)
    batched = batch_size > 1

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = process_batched(_species_probs_as_df)
    else:
        inputs = df.iterrows()
        fn = process_sequentially(_species_probs_as_df)

    with tqdm(total=total, desc="Detecting species...") as pbar:
        for results in run_processing(functools.partial(fn, **kwargs), inputs, num_workers=num_workers):
            yield results
            pbar.update(batch_size)

def embeddings(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Embed a single file
    """
    analyzer = Analyzer()
    return _embed_as_df(analyzer, file_path, **kwargs)

def embeddings_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    """
    Embed all file paths specified in a dataframe
    """
    total = len(df)
    batched = batch_size > 1
    sync = num_workers == 0

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = process_batched(_embed_as_df)
    else:
        inputs = df.iterrows()
        fn = process_sequentially(_embed_as_df)

    with tqdm(total=total, desc="Extracting embeddings...") as pbar:
        for results in run_processing(functools.partial(fn, **kwargs), inputs, num_workers=num_workers):
            yield results
            pbar.update(batch_size)

def embeddings_and_species_probs(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Embed and extract species probabilities for a single file
    """
    analyzer = Analyzer()
    return _embeddings_and_species_probs_as_df(analyzer, file_path, **kwargs)

def embeddings_and_species_probs_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    """
    Embed and extract species probabilities for all file paths in a dataframe
    """
    total = len(df)
    batched = batch_size > 1
    sync = num_workers == 0

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = process_batched(_embeddings_and_species_probs_as_df)
    else:
        inputs = df.iterrows()
        fn = process_sequentially(_embeddings_and_species_probs_as_df)

    with tqdm(total=total, desc="Extracting embeddings and detecting species...") as pbar:
        for results in run_processing(functools.partial(fn, **kwargs), inputs, num_workers=num_workers):
            yield results
            pbar.update(batch_size)

# ------------------------- Single Instance Handlers ------------------------------- #

def _species_probs_as_df(
    analyzer: Analyzer,
    data: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(data.file_path),
        lat=data.get("latitude"),
        lon=data.get("longitude"),
        date=ts.date() if pd.notnull(ts := data.get("timestamp")) else None,
        **kwargs,
    )
    with suppress_output():
        recording.analyze()
    df = pd.DataFrame(recording.detections)
    df["file_path"] = str(data.file_path)
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df

def _embed_as_df(
    analyzer: Analyzer,
    data: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(data.file_path),
        lat=data.get("latitude"),
        lon=data.get("longitude"),
        date=ts.date() if pd.notnull(ts := data.get("timestamp")) else None,
        **kwargs,
    )
    with suppress_output():
        recording.extract_embeddings()
    df = pd.DataFrame([
        pd.concat([
            pd.Series(embedding_info["embeddings"]),
            pd.Series({ k: v for k, v in embedding_info.items() if k != "embeddings" }),
        ])
        for embedding_info in recording.embeddings
    ])
    df["file_path"] = str(data.file_path)
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df

def _embeddings_and_species_probs_as_df(
    analyzer: Analyzer,
    data: pd.Series,
    **kwargs: Any,
) -> pd.DataFrame:
    recording = Recording(
        analyzer,
        str(data.file_path),
        lat=data.get("latitude"),
        lon=data.get("longitude"),
        date=ts.date() if pd.notnull(ts := data.get("timestamp")) else None,
        **kwargs,
    )
    with suppress_output():
        recording.analyze()
        recording.extract_embeddings()
    # map embeddings to a dataframe
    df1 = pd.DataFrame([
        pd.concat([
            pd.Series(embedding_info["embeddings"]),
            pd.Series({ k: v for k, v in embedding_info.items() if k != "embeddings" }),
        ])
        for embedding_info in recording.embeddings
    ]).set_index(["start_time", "end_time"])
    # extract species as columns and merge with embeddings table
    if len(recording.detections):
        df2 = pd.DataFrame(recording.detections).pivot_table(
            index=["start_time", "end_time"],
            columns="label",
            values="confidence",
            fill_value=0,
        )
        # merge with embeddings and fill in missing species probs
        df = df1.merge(df2, left_index=True, right_index=True, how="left").fillna(0)
        # group by multi-index
        df.columns = pd.MultiIndex.from_tuples([
            *[("embedding", i) for i in df1.columns],
            *[("species", i) for i in df2.columns]
        ])
    else:
        df = df1
        # group by multi-index
        df.columns = pd.MultiIndex.from_tuples([
            *[("embedding", i) for i in df1.columns],
        ])
    # make data contiguous
    df._consolidate_inplace()
    # assign primary key
    df["file_path"] = str(data.file_path)
    df["model"] = f"BirdNET_GLOBAL_6K_V{analyzer.version}"
    return df.reset_index()
