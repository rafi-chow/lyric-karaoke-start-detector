## Architecture (high level)

```mermaid
graph TD
  A[Audio Upload] --> B[Feature Extraction<br/>0.5s frames, 80-mel]
  B --> C[Frame Classifier<br/>LogReg + Scaler]
  C --> D[Smoothing + Segmenting]
  D --> E[Karaoke Start Heuristic]
  E --> F[API Response<br/>segments + karaoke_start]
