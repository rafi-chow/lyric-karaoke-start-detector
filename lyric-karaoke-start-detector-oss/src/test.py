from src.inference_baseline import KaraokePredictor

kp = KaraokePredictor(
    model_path=r"C:\Users\psult\lyircs-karoke\models\harmonix_lr.pkl",
    frame_duration=0.5,
    debug = True
)

out = kp.predict(r"C:\Users\psult\lyircs-karoke\src\Justin Bieber - Confident (Audio) ft. Chance The Rapper.wav")

print("Karaoke start:", out["karaoke_start"])
print("\nSegments:")
for seg in out["segments"]:
    dur = seg["end"] - seg["start"]
    print(f"{seg['start']:.1f} → {seg['end']:.1f}  (dur={dur:.1f}s)")
    
