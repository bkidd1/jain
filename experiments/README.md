# Experiments

Results from logit lens and RTP experiments.

## First Run (2026-02-23)

`first_run_results.txt` - Initial logit lens experiments on Llama 3.1 8B

### Key Findings

1. **Multi-hop reasoning is visible in intermediate layers**
   - "Dallas is a city in the state of" → Texas appears at Layer 20 (0.64) and peaks at Layer 24 (1.00)
   - Model correctly retrieves state from city name

2. **Arithmetic knowledge exists internally but doesn't surface**
   - "100 - 37" → Answer "63" appears at Layer 20 (0.55), peaks at Layer 24-28 (0.95-0.98)
   - Final output is just a space — model knows but doesn't say!

3. **Adversarial multi-hop passes**
   - "Largest city in state whose capital is Austin" correctly goes Austin → Texas → Houston
   - Model doesn't fall for surface heuristic (would predict Austin otherwise)

4. **Knowledge retrieval chains work**
   - "Apple was founded by Steve" → "Jobs" at 0.96 confidence by Layer 24

### Implications

The logit lens reveals implicit reasoning that doesn't appear in final outputs.
This is the ground truth signal we'll train the Reasoning Trace Predictor (RTP) to detect.
