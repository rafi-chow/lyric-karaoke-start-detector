# Architecture (high level)

graph TD
  A[Audio Upload]
  B[Feature Extraction<br/>0.5s frames · 80 mel]
  C[Frame Classifier<br/>LogReg + Scaler]
  D[Smoothing & Segmentation]
  E[Karaoke Start Heuristic]
  F[API Response<br/>segments · karaoke_start]

  A --> B --> C --> D --> E --> F

