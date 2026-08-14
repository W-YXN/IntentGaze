# Copyright 2026 Xinan Yan and The Hong Kong University of Science and Technology (Guangzhou)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Canonical grouped-by-subject cross-validation split for IntentGaze.

Fold ``i`` holds out ``CANONICAL_CHUNKS[i]`` as test subjects and uses the
next chunk as validation subjects.  The remaining subjects form the training
set.  Preprocessing imports this module so its fold membership cannot drift
from downstream training/evaluation code.
"""

from __future__ import annotations


CANONICAL_CHUNKS = [
    [13, 15, 21, 24, 26],
    [8, 12, 16, 19, 29],
    [0, 4, 9, 17, 33],
    [1, 2, 5, 11, 27],
    [3, 23, 30, 31, 32],
    [6, 10, 18, 22, 25],
    [7, 14, 20, 28, 34],
]
N_FOLDS = len(CANONICAL_CHUNKS)


def _check_fold_index(fold_index: int) -> int:
    fold_index = int(fold_index)
    if not 0 <= fold_index < N_FOLDS:
        raise ValueError(f"fold_index must be in [0, {N_FOLDS - 1}], got {fold_index}")
    return fold_index


def all_subjects() -> set[int]:
    return {subject for chunk in CANONICAL_CHUNKS for subject in chunk}


def build_fold_split(fold_index: int) -> dict[str, object]:
    """Return the train/validation/test subject sets for one 0-based fold."""
    fold_index = _check_fold_index(fold_index)
    test = set(CANONICAL_CHUNKS[fold_index])
    validation = set(CANONICAL_CHUNKS[(fold_index + 1) % N_FOLDS])
    train = all_subjects() - test - validation
    return {
        "fold_index": fold_index,
        "fold_name": f"fold{fold_index + 1:02d}",
        "train_subject_ids": sorted(train),
        "validation_subject_ids": sorted(validation),
        "test_subject_ids": sorted(test),
        "normalization_fit_subject_ids": sorted(train | validation),
    }
