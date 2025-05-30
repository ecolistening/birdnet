# BirdNet Multiprocessing
A wrapper around `birdnetlib` which has no opinion about directory structure for handling multiprocessing. Robust to failing to load audio.

## Setup
Setup / install dependencies

```sh
uv venv
uv sync
source .venv/bin/activate
```

---

## Usage
First build the file index
```
python main.py build-file-index --audio-dir=/path/to/audio/root/directory \
                                --index-file-name=metadata.parquet
```
> **NB**: Location and date information not yet supported in this function. You can however provide your own index as long as it adheres to the following column requirements
> 1. `uuid` string specifies a unique identifier for each file
> 2. `file_path` string relative to the specified root of all the audio files `--audio-dir`
> 3. `latitude` float (or padded with `nan`)
> 4. `longitude` float (or padded with `nan`)
> 5. `timestamp` datetime (or padded with `nan`)

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

---

## Using Singularity
Build the container:
```sh
singularity build --fakeroot app.sif app.def
```

Run the relevant script within the container:
```sh
singularity run -B /path/to/audio/root/directory:/data app.sif python main.py species-probs --audio-dir=/data --batch-size=6 --save-dir=/data
```
> **NB** If you want a custom save directory, you will need to specify that as a mount point

---

## Tests
Run the tests

```sh
pytest --audio-dir=/path/to/audio/dir
```
