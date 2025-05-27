import logging
import os
from dask.distributed import Client

from birdnetlib.analyzer import Analyzer
from birdnet_multiprocessing.utils import suppress_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_analyzer = None

def get_analyzer(**kwargs):
    global _analyzer
    if _analyzer is None:
        with suppress_output():
            _analyzer = Analyzer(**kwargs)
            logger.info(f"Analyzer loaded in PID {os.getpid()}")
    return _analyzer
