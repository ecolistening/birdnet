from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster

def setup_client(cores: int = 4, memory: int = 8):
    return Client(n_workers=cores, memory_limit=f'{memory}GiB')
