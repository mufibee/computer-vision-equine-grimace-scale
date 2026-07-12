## Final untouched test results

The canonical **resnet18** model was evaluated once on
**916 untouched test images**.

| Metric | Final test result |
|---|---:|
| Weighted cross-entropy loss | 0.9005 |
| Unweighted cross-entropy loss | 0.7680 |
| Accuracy | 0.6572 |
| Macro precision | 0.6099 |
| Macro recall | 0.6025 |
| **Macro F1** | **0.6042** |
| Weighted F1 | 0.6592 |
| Mean prediction confidence | 0.7463 |

The highest per-class recall was obtained for
**No Pain**
(0.7848).

The lowest per-class recall was obtained for
**Severe Pain**
(0.5024).

The model made **314** incorrect predictions.
Of these, **69** had confidence greater than or equal to
80%.

These results are now frozen as the project's final reported test performance.
They must not be used to retrain the model, change preprocessing, tune
hyperparameters, select another checkpoint, or alter decision thresholds.
