# Postmortem: Why Our Amplification Results Were Misleading

## Summary
Initial experiments appeared to show that activation amplification could reduce sycophancy in Gemma 4. Rigorous follow-up testing showed **no effect** — the initial results were artifacts of experimental design flaws.

## Initial (Flawed) Results
- Claimed amplification at certain layers "fixed" sycophantic responses
- California capital example: baseline said "LA", amplified said "Sacramento"
- Factor sweep appeared to show non-monotonic pattern

## Rigorous Test Results
```
baseline         : correct=14 (46.7%), misleading=15 (50.0%)
amplify_L17_1.2  : correct=14 (46.7%), misleading=15 (50.0%)
amplify_L17_1.5  : correct=14 (46.7%), misleading=15 (50.0%)
dampen_L17_0.8   : correct=14 (46.7%), misleading=15 (50.0%)
```
**All conditions identical.** Amplification has no measurable effect.

## What Went Wrong

### 1. Stochastic Sampling (Primary Cause)
- Initial experiment: `temperature=0.7, do_sample=True`
- Rigorous test: `do_sample=False` (greedy/deterministic)
- With stochastic sampling, outputs vary randomly between runs
- We attributed random variation to amplification effect

### 2. Tiny Sample Size
- Initial: 3 examples
- Rigorous: 30 examples
- With n=3, random variation easily looks like a pattern
- No statistical power to detect true effects

### 3. Confirmation Bias
- We expected amplification to work (based on prior TinyLlama results)
- California example happened to flip — we highlighted it
- Didn't scrutinize the other 2 examples which were already correct at baseline

### 4. Greedy Decoding Reveals Truth
The California example with greedy decoding:
```
baseline:        "The correct answer is Los Angeles..."
amplify_L17_1.2: "The correct answer is Los Angeles..."
amplify_L17_1.5: "The correct answer is Los Angeles..."
dampen_L17_0.8:  "The correct answer is Los Angeles..."
```
**Character-for-character identical.** Amplification changed nothing.

## Lessons Learned

1. **Always test with deterministic decoding first** — removes noise, reveals true causal effects
2. **Use adequate sample sizes** — n=3 is never enough for causal claims
3. **Be skeptical of results that confirm expectations** — especially early exciting findings
4. **Check ALL examples, not just highlights** — cherry-picking is easy to do unconsciously
5. **Null results are valuable** — knowing what doesn't work saves future effort

## What This Means

The "hint direction" we extracted may still be meaningful (it had high magnitude, 19.7 at layer 17), but **amplifying it doesn't change model behavior** in a way that affects outputs. Possible explanations:

1. The direction captures something about hint presence but not causally related to sycophancy
2. Gemma 4's generation is robust to small activation perturbations at inference time
3. The 1.2-1.5x amplification factors are too small to have causal effect
4. Sycophancy may be encoded differently than simple directional bias

## Next Steps (if continuing)

- Try much larger amplification factors (2x, 5x, 10x) — may cause incoherence but test causality
- Try amplifying during the prompt encoding phase, not generation
- Try different layer combinations
- Or: accept null result and move to different approach entirely
