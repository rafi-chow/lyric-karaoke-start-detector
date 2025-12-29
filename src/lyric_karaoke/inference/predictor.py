import joblib
import librosa
import numpy as np

from ..features.mel import extract_features
from ..segments.build import predictions_to_segments
from ..segments.clean import clean_segments
from ..segments.smooth import median_smooth, enforce_min_consecutive
from ..karaoke_logic.choose_start import choose_karaoke_start


class KaraokePredictor:
    def __init__(
        self,
        model_path: str,
        frame_duration: float = 0.5,
        thr_segments: float = 0.38,
        thr_start: float = 0.35,
        debug: bool = False,
    ):
        self.model = joblib.load(model_path)
        self.frame_duration = frame_duration
        self.thr_segments = thr_segments
        self.thr_start = thr_start
        self.debug = debug

    def _smooth_binary(self, y: np.ndarray) -> np.ndarray:
        # Robust smoothing: remove flicker + require sustained positives
        y = median_smooth(y, window_size=5)
        y = enforce_min_consecutive(y, min_consecutive=3)  # 3 * 0.5s = 1.5s sustained
        return y

    def _segments_from_probs(
        self,
        probs: np.ndarray,
        times: np.ndarray,
        thr: float,
        *,
        min_duration: float = 1.5,
        merge_gap: float = 1.0,
    ):
        y = (probs >= thr).astype(int)
        y = self._smooth_binary(y)
        raw = predictions_to_segments(times, y, self.frame_duration)
        segs = clean_segments(raw, min_duration=min_duration, merge_gap=merge_gap)
        return y, raw, segs

    def predict(self, audio_path: str):
        # 1) Extract features (must match training feature scale)
        X, times = extract_features(audio_path, self.frame_duration)

        if X.size == 0:
            return {"segments": [], "karaoke_start": None}

        # 2) Probability scores once
        probs = self.model.predict_proba(X)[:, 1]

        # 3) Build displayed segments (precision-oriented)
        y_seg, raw_seg, segments = self._segments_from_probs(
            probs,
            times,
            self.thr_segments,
            min_duration=1.5,
            merge_gap=1.0,
        )

        # 4) Build karaoke-start candidates (recall-oriented, slightly looser)
        y_start, raw_start, start_segments = self._segments_from_probs(
            probs,
            times,
            self.thr_start,
            min_duration=1.0,   # allow earlier “first verse” candidates
            merge_gap=1.0,
        )

        # 5) Song duration (for karaoke logic)
        y_audio, sr = librosa.load(audio_path, sr=None, mono=True)
        song_duration = librosa.get_duration(y=y_audio, sr=sr)

        # 6) Choose karaoke start from start_segments (not the displayed segments)
        karaoke_start = choose_karaoke_start(start_segments, song_duration)

        if self.debug:
            print("X:", X.shape, "times:", times.shape)
            print("X stats:", float(X.min()), float(X.max()), float(X.mean()), float(X.std()))
            print("model:", type(self.model))
            print("prob stats:", float(probs.min()), float(probs.max()), float(probs.mean()))
            print(f"thr_segments={self.thr_segments} ones={int(y_seg.sum())} raw={len(raw_seg)} clean={len(segments)}")
            print(f"thr_start={self.thr_start} ones={int(y_start.sum())} raw={len(raw_start)} clean={len(start_segments)}")
            print("karaoke_start:", karaoke_start)

        return {
            "segments": segments,
            "karaoke_start": karaoke_start,
        }
