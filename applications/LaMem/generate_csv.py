import argparse
import os
import sys

import pandas as pd

if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_paths import CSV_SAVE_DIR, LAMEM_PATH

    # call function by add process_ to the dataset name
    metadata = pd.read_csv("applications/LaMem/LaMem.csv")
    metadata["path"] = metadata["path"].apply(
        lambda x: x.replace("./data/lamem", LAMEM_PATH)
    )
    metadata.to_csv(os.path.join(CSV_SAVE_DIR, "LaMem.csv"), index=False)
    print(
        f"Succesfully processed LaMem dataset with {len(metadata)} images to {CSV_SAVE_DIR}/LaMem.csv"
    )
