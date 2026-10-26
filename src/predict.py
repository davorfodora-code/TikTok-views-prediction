import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Predvidi broj pregleda TikTok videa")
    parser.add_argument("--market", required=True, help='npr. "US/UK/Global", "Brazil/Portugal", "Balkan"')
    parser.add_argument("--language", required=True, help='ISO kod jezika, npr. "en", "es", "sr"')
    parser.add_argument("--followers", type=int, required=True, help="Broj pratilaca autora")
    parser.add_argument("--hour", type=int, required=True, help="Sat objave videa (0-23)")
    parser.add_argument("--day-of-week", type=int, default=2, help="0=ponedeljak ... 6=nedelja")
    parser.add_argument("--hashtag-count", type=int, default=3)
    parser.add_argument("--duration-seconds", type=float, default=30.0)
    args = parser.parse_args()

    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Nema {model_path}. Prvo pokreni: python src/train.py")
    pipeline = joblib.load(model_path)

    row = pd.DataFrame([{
        "market": args.market,
        "language": args.language,
        "author_followers": args.followers,
        "post_hour": args.hour,
        "post_day_of_week": args.day_of_week,
        "hashtag_count": args.hashtag_count,
        "duration_seconds": args.duration_seconds,
    }])

    pred_log = pipeline.predict(row)[0]
    pred_views = np.expm1(pred_log)
    print(f"Predvidjen broj pregleda: {pred_views:,.0f}")


if __name__ == "__main__":
    main()
