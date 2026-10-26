import shutil
import sys
from pathlib import Path

import kagglehub

DATASET = "maratsaratov/tiktok-data"
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    try:
        cache_path = Path(kagglehub.dataset_download(DATASET))
    except Exception as e:
        sys.exit(
            "Preuzimanje nije uspelo. Proveri da li imas Kaggle API token u "
            "~/.kaggle/access_token (noviji KGAT_ token) ili ~/.kaggle/kaggle.json "
            f"(stari username+key token).\n\nGreska: {e}"
        )

    csv_files = list(cache_path.glob("*.csv"))
    if not csv_files:
        sys.exit(f"Download zavrsen ali nema .csv fajlova u {cache_path}.")

    for f in csv_files:
        shutil.copy(f, RAW_DIR / f.name)

    print(f"Preuzeto u {RAW_DIR}:")
    for f in csv_files:
        print(" -", f.name)


if __name__ == "__main__":
    main()
