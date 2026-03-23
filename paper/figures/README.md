# Figures

This directory should contain:

1. **phase_transition.png** - Bar chart showing AUROC vs number of training architectures
   - X-axis: "Same model", "1 foreign", "2 foreign", "3 models"
   - Y-axis: AUROC on TinyLlama-Chat (0.5 to 1.0)
   - Key visual: sharp jump from ~0.65 to ~0.93 at 2 architectures

2. **scale_transfer.png** - Line plot showing AUROC across model scales
   - X-axis: Model size (1.5B, 7B, 14B)
   - Y-axis: AUROC (0.85 to 1.0)
   - Include error bars from bootstrap resampling

3. **unfaithful_example.png** - Diagram showing logit lens analysis
   - Left: Prompt with misleading hint
   - Middle: Layer-by-layer top tokens (heatmap style)
   - Right: Generated CoT that doesn't mention hint
   - Highlight the layers where hint token appears

## Generating Figures

```python
# See notebooks/generate_figures.ipynb for code to generate these figures
```
