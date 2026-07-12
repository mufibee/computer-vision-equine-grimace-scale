# Phase 9 Critical Analysis

## Validation behaviour

The final selected model achieved validation accuracy of
0.7905 and validation Macro F1 of
0.7077 across 315 images.

The strongest class by validation class accuracy was
**No Pain**
(0.8818). The weakest class was
**Severe Pain**
(0.6000).

The model made 66 validation errors.
The most frequent error was No Pain being predicted as Moderate Pain (18 images).

## Confidence

Mean confidence was 0.8664 for correct
predictions and 0.6727 for incorrect
predictions. The difference was +0.1938.

There were 19 mistakes with confidence greater than or
equal to 80%. Such errors are especially important
because they show that confidence should not be interpreted as guaranteed correctness.

## Learned filters and feature maps

The first-layer visualizations show the small edge, colour-contrast, and intensity
patterns learned by the initial convolution. Early feature maps retain relatively
fine spatial detail. Middle feature maps combine these responses into more structured
textures and shapes. Deep feature maps are lower resolution and more selective,
reflecting the hierarchical feature extraction discussed in CNN lectures.

The visualizations should be interpreted cautiously. A bright activation indicates
that one channel responded strongly in that location. It does not establish that the
region caused the final classification, and feature maps alone cannot prove that the
network learned clinically meaningful equine pain cues.

## Limitations

1. The validation split is finite, so class-level conclusions may be sensitive to a
   small number of examples.
2. Pain classes can share subtle facial characteristics, making adjacent classes
   difficult to separate.
3. Background, crop position, lighting, horse identity, and image quality may influence
   activations.
4. Feature-map inspection is qualitative and channel selection affects what is shown.
5. The model predicts image-level pain categories and does not independently measure
   ears, eyes, muzzle, nostrils, or cheek tension.
6. No conclusions from this notebook are based on the untouched test split.

## Methodological conclusion

Phase 9 explains model behaviour using validation data without changing the trained
model or selecting a new configuration. Phase 10 must now perform the first and only
evaluation of this frozen final model on the untouched test set.