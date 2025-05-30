import datetime as dt
import pathlib
import pandas as pd
import re
import logging
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Callable, ClassVar

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

AUDIO_FILE_REGEX = re.compile(r".*\.(wav|flac|mp3)$", re.IGNORECASE)

class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)

@dataclass
class Object(ABC):
    @property
    def fields(self):
        return [f.name for f in fields(self)]

    @classproperty
    def fields(cls):
        return [f.name for f in fields(cls)]

@dataclass
class File(Object):
    file_id: str
    file_name: str
    file_path: str | pathlib.Path
    timestamp: dt.datetime

@dataclass
class Site(Object):
    site_id: str
    site_name: str
    latitude: float
    longitude: float

@dataclass
class Dataset(Object):
    audio_dir: str | pathlib.Path

    _FILE_INDEX: ClassVar[str] = "files.parquet"
    _LOCATIONS_INDEX: ClassVar[str] = "locations.parquet"

    def __post_init__(self) -> None:
        self.files = self._load_files()
        self.metadata = self._load_metadata()

    @abstractmethod
    def to_datetime(self, file_name):
        pass

    @abstractmethod
    def to_site_name(self, file_path):
        pass

    def _load_files(self):
        if (self.audio_dir / self._FILE_INDEX).exists():
            return pd.read_parquet(self.audio_dir / self._FILE_INDEX)
        files = pd.DataFrame([
            File(
                file_path=str(file_path),
                file_id=str(uuid.uuid4()),
                file_name=file_path.name,
                timestamp=self.to_datetime(file_path.name)
            )
            for file_path in pathlib.Path(self.audio_dir).rglob('*')
            if AUDIO_FILE_REGEX.match(str(file_path))
        ])
        files.to_parquet(self.audio_dir / self._FILE_INDEX)
        return files

    @abstractmethod
    def _load_locations(self, file_path):
        pass

    def _load_metadata(self):
        # attach foreign key for site
        self.files["site_name"] = self.files["file_path"].map(self.to_site_name)
        # load site information
        self.locations = self._load_locations()
        return self.files.merge(
            self.locations[Site.fields],
            left_on="site_name",
            right_on="site_name",
        )
