"""Load the datasets."""

import os
import re

from .settings import settings


def load_20_mille_lieues_sous_les_mers() -> str:
    """Load the book "Vingt mille lieues sous les mers."""
    with open(
        os.path.join(
            settings.DATASET_FOLDER, "Vingt_mille_lieues_sous_les_mers_Texte_entier.txt"
        ),
        "r",
    ) as file:
        content = file.read()
    return re.sub(r"\n\n+", "\n", content)


if __name__ == "__main__":
    print(load_20_mille_lieues_sous_les_mers()[:400])
