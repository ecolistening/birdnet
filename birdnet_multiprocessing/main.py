import argparse
import dask
import dask.dataframe as dd
import datetime as dt
import pathlib
import pandas as pd
import logging

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from collections import defaultdict
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Tuple,
)

from birdnet_multiprocessing.analyzer import get_analyzer
from birdnet_multiprocessing.utils import suppress_output

def birdnet_column_metadata():
    return pd.DataFrame({
        "file_path": pd.Series(dtype="object"),
        "common_name": pd.Series(dtype="object"),
        "scientific_name": pd.Series(dtype="object"),
        "label": pd.Series(dtype="object"),
        "confidence": pd.Series(dtype="float64"),
        "start_time": pd.Series(dtype="float64"),
        "end_time": pd.Series(dtype="float64"),
    })

def species_presence_probs(
    df: pd.DataFrame,
    analyzer_kwargs: Dict,
    **kwargs: Any,
) -> pd.DataFrame:
    columns = birdnet_column_metadata().columns
    analyzer = get_analyzer(**analyzer_kwargs)
    results = []
    for i, file_info in df.iterrows():
        recording = Recording(
            analyzer,
            file_info.file_path,
            lat=file_info.latitude,
            lon=file_info.longitude,
            date=file_info.timestamp.date(),
            **kwargs,
        )
        with suppress_output():
            recording.analyze()
        result_df = pd.DataFrame(recording.detections)
        result_df["file_path"] = file_info.file_path
        results.append(result_df)
    if results:
        return pd.concat(results, axis=0)[columns]
    else:
        return pd.DataFrame(columns=columns)

def species_presence_probs_multiprocessing(df: pd.DataFrame, num_partitions: int = 4):
    client = Client()

    analyzer = Analyzer()
    analyzer_kwargs = dict(
        classifier_labels_path=analyzer.classifier_labels_path,
        classifier_model_path=analyzer.classifier_model_path,
    )

    result_df = (
        dd.from_pandas(df.reset_index(drop=True), npartitions=num_partitions)
        .map_partitions(
            species_presence_probs,
            analyzer_kwargs=analyzer_kwargs,
            meta=birdnet_column_metadata()
        )
    ).compute()

    client.close()

    return result_df
