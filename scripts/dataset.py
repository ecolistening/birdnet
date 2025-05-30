import pathlib
import pandas as pd
import re
import logging

from abc import ABC, abstractmethod
from typing import Callable

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

class Dataset(ABC):
    _FILE_INDEX: str = "file_index.parquet"
    _LOCATIONS_INDEX: str = "locations.parquet"

    def __init__(self, audio_dir: pathlib.Path) -> None:
        self.audio_dir = audio_dir

        self.files = self._load_file_index()
        self.metadata = self._load_metadata()

    @abstractmethod
    def to_datetime(self, file_name):
        pass

    @abstractmethod
    def to_site_name(self, file_path):
        pass

    def _load_file_index(self):
        if (self.audio_dir / self._FILE_INDEX).exists():
            return pd.read_parquet(self.audio_dir / self._FILE_INDEX)
        else:
            return self._build_file_index()

    def _load_metadata(self):
        if not (self.audio_dir / self._LOCATIONS_INDEX).exists():
            log.warning("'locations.parquet' is missing, continuing without site level information")
            return self.files
        else:
            self.locations = pd.read_parquet(self.audio_dir / self._LOCATIONS_INDEX)
            files["site"] = self.files["file_path"].map(self.to_site_name)
            return files.merge(
                self.locations[["site", "latitude", "longitude"]],
                left_on="site",
                right_on="site",
            )

    def _build_file_index(self):
        files = pd.DataFrame([
            dict(
                file_path=str(file_path),
                uuid=str(uuid.uuid4()),
                file_name=file_path.name,
                timestamp=self.to_datetime(file_path.name)
            )
            for file_path in pathlib.Path(audio_dir).rglob('*')
            if AUDIO_FILE_REGEX.match(str(file_path))
        ])
        files.to_parquet(self.audio_dir / self._FILE_INDEX)
        return files
