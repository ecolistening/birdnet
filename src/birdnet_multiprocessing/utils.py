import contextlib
import os
import sys

from typing import Any, List

@contextlib.contextmanager
def suppress_output():
    with open(os.devnull, 'w') as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield

def chunked(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items.iloc[i:i + batch_size]
