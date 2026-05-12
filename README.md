# IntentGaze Code

Data-processing pipeline and training scripts accompanying the IntentGaze paper.
The dataset is released separately and is **not** included in this repository.

## Data

The dataset is hosted on OSF (anonymous view-only link for blind review):

<https://osf.io/tfjgu/overview?view_only=41d07c32210741c6995b99b551e7171e>

Per-frame schema, directory layout, and anonymization notes are documented in the
`README.md` on OSF. The dataset is published as a split ZIP archive; reconstruct
and extract it as follows:

```bash
cat Dataset.zip.001 Dataset.zip.002 > Dataset.zip
unzip Dataset.zip
```

This produces a `Dataset/` folder organized by `<context>/<task>/...`.

## Layout

For zero-argument execution, place the extracted dataset under this directory so
that the paths match each script's default `--input-dir`:

```
Code/
├── 01-resample_dataset.py
├── 02-split_sessions.py
├── 03-process_pitch_yaw.py
├── 04-process_to_npy.py
├── 05-train_regression.py
├── 06-train_saccade_detector.py
├── requirements.txt
└── Dataset/              <-- extract the OSF archive directly here
    ├── Lying/
    ├── Sitting/
    └── ...
```

To use any other location, pass `--input-dir <path>` to `01-resample_dataset.py`.

## Environment

Python 3.10 or later. Install dependencies with:

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended for the two training scripts (05 and 06); the
processing scripts (01-04) run on CPU.

## Pipeline

```
raw CSV (from OSF)
 └─ 01-resample_dataset.py        → resampled_csv/
     └─ 02-split_sessions.py      → session_csv/, session_csv_transition/
         └─ 03-process_pitch_yaw.py → processed_csv/, processed_csv_transition/
             └─ 04-process_to_npy.py → npy_data/, npy_data_transition/
                 ├─ 05-train_regression.py        → trained_regression/
                 └─ 06-train_saccade_detector.py  → trained_saccade_detector/
```

## Usage

Run the scripts in order. With the default layout above, stages 01-04 require no
arguments:

```bash
python 01-resample_dataset.py
python 02-split_sessions.py
python 03-process_pitch_yaw.py
python 04-process_to_npy.py
python 05-train_regression.py       --epochs 160
python 06-train_saccade_detector.py --stage1-epochs 80 --stage2-epochs 40
```

The epoch counts above match those reported in the paper. All other
hyperparameters (batch size, learning rates, weight decay, focal-loss `gamma`/
`alpha`, cosine-annealing schedule, dropout) are encoded as the default values
of the corresponding CLI flags, so no additional arguments are needed for
reproduction. The default `--seed 42` is used throughout. Run any script with
`--help` for the full list of flags.

The training scripts save a checkpoint for every epoch
(`epoch_XXX.pt`, `stageN_epoch_XXX.pt`); per-epoch validation selection is
performed externally and is not part of these scripts.

## Outputs

Each stage writes a manifest CSV (or summary JSON) alongside its data outputs.
The training scripts produce per-epoch checkpoints under their output
directories:

- `trained_regression/epoch_XXX.pt`
- `trained_saccade_detector/stage1_epoch_XXX.pt` and `stage2_epoch_XXX.pt`

Selecting the checkpoint to evaluate on the held-out test split (e.g. by the
minimum loss on a validation split) is performed externally; the training
scripts intentionally do not write a single "final" or "best" file.

---

License and citation will be added after the blind-review period.
