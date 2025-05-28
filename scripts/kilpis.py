import datetime as dt
import logging
import pandas as pd
import pathlib

from birdnetlib import Analyzer, Recording
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer

logger =  logging.getLogger(__name__)

class SM4:
    def summary(self, file_path):
        metadata = pd.read_csv(file_path) # TODO: switch with robust parse summary file text
        recorder_id = file_path.name.split("_")[0]
        latitude = float(summary_metadata["LAT"].unique()[0])
        longitude = float(summary_metadata["LON"].unique()[0])
        return dict(recorder=recorder_id, latitude=latitude, longitude=longitude)

def try_or(func: Callable, default: Any) -> Any:
    try:
        return func()
    except Exception as e:
        return default

class KilpisDataset:
    _IGNORE_DIRS = ["DOCS_ANALYSIS"]

    def __init__(root_dir: Path):
        self.root_dir = root_dir

    def recorder_directories(self):
        return pathlib.Path(root_dir).glob('K*'))

    def recorder_locations(self) -> pd.DataFrame:
        recorder_type = SM4()
        locations = []
        for path in self.recorder_directories:
            summary_path = try_or(lambda: list(pathlib.Path(path).glob("*_Summary.txt"))[0], None)
            assert summary_path is not None, f"summary file missing for recorder {path}, no location information available"
            summary_metadata = recorder_type.summary(summary_path)
            locations.append(summary_metadata)
        return pd.DataFrame(locations)

    def recorder_directory_to_birdnet_predictions(
        self,
        directory: str | pathlib.Path,
        save_path: str | pathlib.Path,
        date: dt.datetime,
        **kwargs: Any,
    ) -> None:
        analyzer = Analyzer()
        recorder_kwargs = dict(lon=longitude, lat=latitude, date=date)
        batch = DirectoryMultiProcessingAnalyzer(directory, analyzers=[analyzer], **recorder_kwargs, **kwargs)

        def on_complete_callback(recordings: List[Recording]):
            results = []
            for recording in recordings:
                if recording.error:
                    logging.warning("Failed to collect results for {recording.path}")
                    logging.error(recording.error_message)
                else:
                    recording_df = pd.DataFrame(recording.detections)
                    recording_df["file_path"] = recording.path
                    recording_df["latitude"] = latitude
                    recording_df["longitude"] = longitude
                    results.append(recording_df)
            df = pd.concat(results, axis=0)
            df.to_parquet(save_path)

        batch.on_analyze_directory_complete = on_complete_callback
        batch.process()
