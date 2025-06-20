FROM python:3.10-slim

ENV PATH="/root/.local/bin:/cargo/bin:$PATH"
ENV DATA_PATH=/data
ENV PYTHONPATH=/app/src
ENV VIRTUAL_ENV_PATH=/app/.venv

RUN apt-get update
RUN apt-get install --no-install-recommends -y curl ffmpeg
RUN  apt-get clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /cargo
RUN curl -LsSf https://astral.sh/uv/install.sh | CARGO_HOME=/cargo sh

WORKDIR /app

COPY uv.lock pyproject.toml /app/
COPY src /app/src
COPY main.py /app/main.py

RUN uv venv $VIRTUAL_ENV_PATH
RUN . $VIRTUAL_ENV_PATH/bin/activate
RUN uv sync --no-cache --link-mode=copy

ENTRYPOINT ["/bin/bash", "-c", "cd /app && . /app/.venv/bin/activate && uv --no-cache run \"$@\"", "--"]
