from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 42  # deterministicki rezultati langdetect-a

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

LANGUAGE_TO_MARKET = {
    "en": "US/UK/Global",
    "es": "LATAM/Spain",
    "pt": "Brazil/Portugal",
    "id": "Indonesia",
    "th": "Thailand",
    "vi": "Vietnam",
    "ar": "MENA",
    "fr": "France/Francophone",
    "de": "Germany/DACH",
    "ru": "Russia/CIS",
    "tr": "Turkey",
    "ja": "Japan",
    "ko": "Korea",
    "hi": "India",
    "it": "Italy",
    "pl": "Poland",
    "sr": "Balkan",
    "hr": "Balkan",
    "bs": "Balkan",
}

FEATURE_COLUMNS = [
    "market", "language", "author_followers", "post_hour",
    "post_day_of_week", "hashtag_count", "duration_seconds", "views",
]


def detect_language(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def load_raw() -> pd.DataFrame:
    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"Nema CSV fajlova u {RAW_DIR}. Prvo pokreni: python src/download_data.py"
        )
    return pd.read_csv(csv_files[0])


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["text_part"] = df["text_part"].fillna("")
    df["hashtags"] = df["hashtags"].fillna("")

    df["language"] = df["text_part"].apply(detect_language)
    df["market"] = df["language"].map(LANGUAGE_TO_MARKET).fillna("Other")

    df["human_time"] = pd.to_datetime(df["human_time"], errors="coerce")
    df["post_hour"] = df["human_time"].dt.hour
    df["post_day_of_week"] = df["human_time"].dt.dayofweek

    df["author_followers"] = pd.to_numeric(df["author_followers"], errors="coerce")
    df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce")
    df["views"] = pd.to_numeric(df["views"], errors="coerce")

    df["hashtag_count"] = df["hashtags"].apply(
        lambda s: len([h for h in str(s).split(",") if h.strip()])
    )

    df = df[FEATURE_COLUMNS].dropna(subset=["views", "author_followers", "post_hour"])
    return df


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    print(f"Ucitano {len(raw)} redova. Kolone: {list(raw.columns)}")

    features = build_features(raw)
    out_path = PROCESSED_DIR / "features.csv"
    features.to_csv(out_path, index=False)

    print(f"Sacuvano {len(features)} redova u {out_path}")
    print(features.head())


if __name__ == "__main__":
    main()
