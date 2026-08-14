# IntentGaze Code

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)

Data-processing pipeline and training scripts accompanying the IntentGaze paper.
The accompanying dataset is distributed separately through Zenodo.

## Data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21933665.svg)](https://doi.org/10.5281/zenodo.21933665)

Per-frame schema, directory layout, and anonymization notes are documented in the
dataset metadata supplied with the Zenodo archive. The dataset is published as a split ZIP archive; reconstruct
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
└── Dataset/              <-- extract the Zenodo archive directly here
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
raw CSV (from Zenodo)
 └─ 01-resample_dataset.py        → resampled_csv/
     └─ 02-split_sessions.py      → session_csv/, session_csv_transition/
         └─ 03-process_pitch_yaw.py → processed_csv/, processed_csv_transition/
             └─ 04-process_to_npy.py → npy_data_fold01/ … npy_data_fold07/
                                      → npy_data_transition_fold01/ … npy_data_transition_fold07/
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
python 05-train_regression.py       --fold-index 0 --epochs 160
python 06-train_saccade_detector.py --fold-index 0 --stage1-epochs 80 --stage2-epochs 40
```

Stage 04 processes all seven subject-held-out folds by default.  Each fold has
its own normalization parameters, and `--fold-index` makes each training script
select the matching NPY directories, normalization parameters, and training
subjects.  Use `--global-normalization-all-csv` on stage 04 only when an
unsplit, all-CSV normalization run is explicitly intended.

The epoch counts above match those reported in the paper. All other
hyperparameters (batch size, learning rates, weight decay, focal-loss `gamma`/
`alpha`, cosine-annealing schedule, dropout) are encoded as the default values
of the corresponding CLI flags, so no additional arguments are needed for
reproduction. The default `--seed 42` is used throughout. Run any script with
`--help` for the full list of flags.

The training scripts save a checkpoint for every epoch
(`epoch_XXX.pt`, `stageN_epoch_XXX.pt`). Per-epoch validation and checkpoint
selection are performed externally.

## Outputs

Each stage writes a manifest CSV (or summary JSON) alongside its data outputs.
The training scripts produce per-epoch checkpoints under their output
directories:

- `trained_regression/epoch_XXX.pt`
- `trained_saccade_detector/stage1_epoch_XXX.pt` and `stage2_epoch_XXX.pt`

Selecting the checkpoint to evaluate on the held-out test split (e.g. by the
minimum loss on a validation split) is performed externally, with all per-epoch
checkpoints retained for reproducibility.

## License

This code is licensed under the [Apache License 2.0](LICENSE).

## Citation

Please cite the associated IntentGaze paper when using this code. For direct dataset reuse, cite:

```bibtex
@dataset{hu2026_IntentGazeDataset,
  author = {Hu, Xuning and Yan, Xinan and Zhang, Yichuan and Wei, Yushi and Li, Yue and Stuerzlinger, Wolfgang and Liang, Hai-Ning},
  title = {IntentGaze Dataset: Official Data for "IntentGaze: Task-Aligned Gaze Correction for Stable and Responsive Gaze Interaction in XR"},
  month = aug,
  year = 2026,
  publisher = {Zenodo},
  version = {1.0.0},
  doi = {10.5281/zenodo.21933665},
  url = {https://doi.org/10.5281/zenodo.21933665},
}
```
