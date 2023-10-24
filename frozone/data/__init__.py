import os
from glob import glob as glob  # glob
from typing import List, Tuple, Optional

import numpy as np


# X, U, S, R
DataSequence = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
Dataset = List[DataSequence]
# X, U, S, R, Z
DataSequenceSim = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
DatasetSim = List[DataSequenceSim]

TRAIN_TEST_SPLIT = 0.9
RAW_SUBDIR = "Raw"
PROCESSED_SUBDIR = "Processed"
TRAIN_SUBDIR = "Train"
TEST_SUBDIR = "Test"

PHASES = {
    1: "Inactive",
    2: "All_Ready",
    4: "Preheating",
    8: "Drop",  # Automated growth starts here
    16: "Snoevs",
    32: "Vending",
    64: "Necking",  # This is where it is interesting
    128: "Unused",  # Necking and PreCone can be combined, as the split is artificial
    256: "PreCone",  # This is where it is also interesting
    512: "Cone",  # Measurement jump when this phase starts
    1024: "Full_Diameter",  # Stops being interesting here
    2048: "Fast_Closing",  # This one or closing
    4096: "Closing",
    8192: "Cooling",
}
# Use indices
PHASE_TO_INDEX = { 8: 0, 16: 1, 32: 2, 64: 3, 256: 4, 512: 5 }

def list_processed_data_files(data_path: str, train_test_subdir: str) -> list[str]:
    path = os.path.join(data_path, PROCESSED_SUBDIR, train_test_subdir, "**", "*.npz")
    return glob(path, recursive=True)
