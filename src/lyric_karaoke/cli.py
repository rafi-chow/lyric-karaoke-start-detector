import argparse
import json
from lyric_karaoke.inference.predictor import KaraokePredictor


def main():
    p = argparse.ArgumentParser(description="Predict karaoke start + lyric segments for an audio file.")
    p.add_argument("--model", required=True, help="Path to a trained .pkl model (joblib).")
    p.add_argument("--audio", required=True, help="Path to an audio file (wav/mp3).")
    p.add_argument("--frame-duration", type=float, default=0.5)
    p.add_argument("--thr-segments", type=float, default=0.40)
    p.add_argument("--thr-start", type=float, default=0.35)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    pred = KaraokePredictor(
        model_path=args.model,
        frame_duration=args.frame_duration,
        thr_segments=args.thr_segments,
        thr_start=args.thr_start,
        debug=args.debug,
    )
    out = pred.predict(args.audio)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
