# BirdNet
A wrapper around `birdnetlib` useful for use on HPCs. This package has no opinion about directory structure when ingesting audio, it builds or uses an existing file index (as `.csv`, `.parquet`, or other, see the [`load_file_metadata` function](https://github.com/ecolistening/birdnet/blob/1.1.0/src/cli/utils.py#L14)) to find audio files.

 Each row in the index file must follow the following specification:

| Field | Type | Required | Description |
| ----- | ---- | -------- | ----------- |
| `file_id` | string | yes | a unique identifier for each file such as a UUID or integer index |
| `file_path` | string | yes | the file path relative to the project root directory of audio files |
| `latitude` | float | no | used when provided in BirdNET predictions (padded with `nan` permitted) |
| `longitude` | float | no | used when provided in BirdNET predictions (padded with `nan` permitted) |
| `timestamp` | datetime64 | no | used when provided in BirdNET predictions (padded with `nan` permitted) |

`file_path` must be specified relative to the `--audio-dir` passed at runtime.

A helper function is provided to build the file index. We do not currently support attaching location or datetime information due to differences in the way these might be saved. If you want to leverage these fields, you will need to attach them yourself to the built file index or provide your own.

## Setup
There are two ways to setup the package:

1. Installing locally using `uv`
2. Building inside a container using `singularity`

### UV
First install [here](https://docs.astral.sh/uv/getting-started/installation/).

Next create and source a virtual environment and then install the package dependencies:

```sh
uv venv
source .venv/bin/activate
uv sync
```

### Singularity Setup
Build the container:
```sh
singularity build --fakeroot app.sif app.def
```

Check python runs correctly within the container:
```
singularity run app.sif uv run python
```

Run the relevant script within the container:
```sh
singularity run -B /path/to/audio/root/directory:/data app.sif python main.py species-probs --audio-dir=/data --batch-size=6 --save-dir=/data
```

> **NB** If you want to use a custom save directory, you will need to specify that as a mount point

### Slurm
Scripts using singularity are provided for use on a HPC.

- First run the build steps above using singularity.
- Next create a `.env` file with environment variables as per `.env.template`.

Scripts to append jobs to the queue using environment variables set in `.env` are in `slurm/`.

```sh
./slurm/species_probs.sh
./slurm/embed.sh
```

---

## Usage
First build the file index
```
python main.py index-files --audio-dir=/path/to/audio/root/directory \
                           --index-file-name=metadata.parquet
```

An index of files found to be invalid (failed to load the audio) will be saved to `audio_dir / "failed_files.parquet"`.

Extract species probabilities for each file:
```sh
python main.py species-probs --audio-dir=/path/to/audio/root/directory \
                             --index-file-name=metadata.parquet \
                             --batch-size=6 \
                             --save-dir /home/m4gpie/data/
```
> **NB**: Files with no detected species are missing from the results

Extract feature embeddings:
```sh
python main.py embed --audio-dir=/path/to/audio/root/directory \
                     --index-file-name=metadata.parquet \
                     --batch-size=6 \
                     --save-dir /home/m4gpie/data/
```
> **NB**: each row in the resulting table(s) correspond to 1024 features corresponding to a 3s frame

Extract both embeddings and species:
```sh
python main.py embeddings-and-species-probs --audio-dir=/path/to/audio/root/directory \
                                            --index-file-name=metadata.parquet \
                                            --batch-size=6 --save-dir /home/m4gpie/data/
```
> **NB**: Embeddings includes all file segments, which may have no species predictions. All species are padded with zero probabilities for files with non-detections
> Resulting file size may be larger, however this also accounts for species absence in predicted probabilities which can be useful for analyses

Note that the `--batch-size` flag is only for batching audio files for each worker process to handle, within which each files are handled sequentially, rather than batch processing on the GPU. `birdnetlib` does not support batch evaluation on the GPU.

---

## Tests
Run the tests

```sh
pytest --audio-dir=/path/to/audio/dir
```
