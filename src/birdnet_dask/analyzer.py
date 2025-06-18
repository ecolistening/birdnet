from birdnetlib.analyzer import Analyzer
from birdnet_dask.utils import suppress_output

_analyzer = None

@suppress_output()
def _fetch_analyzer():
    global _analyzer
    if _analyzer is None:
        _analyzer = Analyzer()
    return _analyzer
