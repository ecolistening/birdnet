import argparse
import functools
import multiprocessing as mp
import pathlib
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

from birdnet_multiprocessing.utils import chunked, suppress_output, try_or

__ALL__ = [
    "species_probs",
    "species_probs_multiprocessing",
    "embeddings",
    "embeddings_multiprocessing",
]

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def species_probs(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = Analyzer()
    return _species_probs_as_df(analyzer, file_path, **kwargs)

def species_probs_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    total = len(df)
    batched = batch_size > 1
    sync = num_workers == 0

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = _batch_species_probs_from_audio_files
    else:
        inputs = df.iterrows()
        fn = _species_probs_from_audio_file

    with tqdm(total=total, desc="Analysing...") as pbar:
        if sync:
            for results in handle_processing(functools.partial(fn, **kwargs), inputs):
                yield results
                pbar.update(batch_size)
        else:
            pool_kwargs = dict(processes=num_workers, initializer=_init_worker)
            for results in handle_multiprocessing(functools.partial(fn, **kwargs), inputs, **pool_kwargs):
                yield results
                pbar.update(batch_size)

def embeddings(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = Analyzer()
    return _embed_as_df(analyzer, file_path, **kwargs)

def embeddings_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    total = len(df)
    batched = batch_size > 1
    sync = num_workers == 0

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = _batch_embed_audio_files
    else:
        inputs = df.iterrows()
        fn = _embed_audio_file

    with tqdm(total=total, desc="Analysing...") as pbar:
        if sync:
            for results in handle_processing(functools.partial(fn, **kwargs), inputs):
                yield results
                pbar.update(batch_size)
        else:
            pool_kwargs = dict(processes=num_workers, initializer=_init_worker)
            for results in handle_multiprocessing(functools.partial(fn, **kwargs), inputs, **pool_kwargs):
                yield results
                pbar.update(batch_size)

def embeddings_and_species_probs(
    file_path: str,
    **kwargs: Any,
) -> pd.DataFrame:
    analyzer = Analyzer()
    return _embeddings_and_species_probs_as_df(analyzer, file_path, **kwargs)

def embeddings_and_species_probs_multiprocessing(
    df: pd.DataFrame,
    num_workers: int,
    batch_size: int = 1,
    **kwargs: Any,
) -> Iterable[pd.Series]:
    total = len(df)
    batched = batch_size > 1
    sync = num_workers == 0

    if batched:
        inputs = list(chunked(df, batch_size))
        fn = _batch_embeddings_and_species_probs_from_audio_files
    else:
        inputs = df.iterrows()
        fn = _embeddings_and_species_probs_from_audio_file

    with tqdm(total=total, desc="Analysing...") as pbar:
        if sync:
            for results in handle_processing(functools.partial(fn, **kwargs), inputs):
                yield results
                pbar.update(batch_size)
        else:
            pool_kwargs = dict(processes=num_workers, initializer=_init_worker)
            for results in handle_multiprocessing(functools.partial(fn, **kwargs), inputs, **pool_kwargs):
                yield results
                pbar.update(batch_size)

# --------------------------------------------------------------- #

analyzer = None
@suppress_output()
def _init_worker():
    """
    Instantiate a single analyzer on each worker, cached globally.
    """
    global analyzer
    analyzer = Analyzer()

def handle_processing(fn: Callable, inputs: Iterable):
    _init_worker()
    log.info("Synchronous processing")
    return map(fn, inputs)

def handle_multiprocessing(fn: Callable, inputs: Iterable, **kwargs: Any):
    log.info("Concurrent processing")
    with mp.Pool(**kwargs) as map_pool:
        for results in map_pool.imap_unordered(fn, inputs):
            yield results

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
        date=try_or(lambda: data.get("timestamp", None).date(), None),
        **kwargs,
    )
    with suppress_output():
        recording.analyze()
    df = pd.DataFrame(recording.detections)
    df["file_path"] = str(data.file_path)
    return df

def _batch_species_probs_from_audio_files(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return pd.concat([_species_probs_as_df(analyzer, row) for i, row in df.iterrows()], axis=0)

def _species_probs_from_audio_file(item: Tuple[int, pd.Series], **kwargs: Any) -> pd.DataFrame:
    global analyzer
    i, row = item
    return _species_probs_as_df(analyzer, row, **kwargs)

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
        date=try_or(lambda: data.get("timestamp", None).date(), None),
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
    return df

def _batch_embed_audio_files(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return pd.concat([_embed_as_df(analyzer, row) for i, row in df.iterrows()], axis=0)

def _embed_audio_file(item: Tuple[int, pd.Series], **kwargs: Any) -> pd.DataFrame:
    global analyzer
    i, row = item
    return _embed_as_df(analyzer, row, **kwargs)

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
        date=try_or(lambda: data.get("timestamp", None).date(), None),
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
    df._consolidate_inplace()
    # assign primary key
    df["file_path"] = str(data.file_path)
    return df.reset_index()

def _batch_embeddings_and_species_probs_from_audio_files(df: pd.DataFrame, **kwargs: Any) -> pd.DataFrame:
    return pd.concat([
        _embeddings_and_species_probs_as_df(analyzer, row) for i, row in df.iterrows()
    ], join="outer").fillna(0)

def _embeddings_and_species_probs_from_audio_file(item: Tuple[int, pd.Series], **kwargs: Any) -> pd.DataFrame:
    global analyzer
    i, row = item
    return _embeddings_and_species_probs_as_df(analyzer, row, **kwargs)
