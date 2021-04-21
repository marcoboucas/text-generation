"""Settings for the data processing / loading."""

import os


class settings:
    ROOT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    DATASET_FOLDER = os.path.join(ROOT_FOLDER, "dataset")
