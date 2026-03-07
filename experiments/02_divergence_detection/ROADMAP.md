# Research Roadmap: Divergence Detection

## Current Status

**Key Finding:** Cross-architecture training outperforms same-model training for unfaithfulness detection.
- Detector trained on Qwen+Phi-2 achieves 0.928 AUROC on TinyLlama
- Detector trained on TinyLlama achieves only 0.746 AUROC on TinyLlama
- Delta: +18.2 percentage points

## Research Directions

### 1. Scaling Laws (ACTIVE)
**Question:** How does architectural diversity affect detection performance?

**Experiments:**
- [ ] Train on 1 model → test on TinyLlama
- [ ] Train on 2 models → test on TinyLlama  
- [x] Train on 2 models (Qwen+Phi-2) → 0.928 AUROC ✅
- [ ] Train on 3 models → test on TinyLlama
- [ ] Train on 4+ models → test on TinyLlama
- [ ] Plot: # architectures vs AUROC curve

**Hypothesis:** More architectural diversity → better generalization, possibly with diminishing returns.

---

### 2. Mechanistic Analysis
**Question:** Why does cross-architecture training work better?

**Experiments:**
- [ ] Probe detector internal representations (v1 vs v3)
- [ ] Compare attention patterns between detectors
- [ ] Identify which features v3 learns that v1 misses
- [ ] Ablation: which training examples matter most?

**Hypothesis:** Cross-architecture training forces learning of architecture-agnostic unfaithfulness signatures.

---

### 3. Feature Analysis
**Question:** What makes unfaithfulness detectable?

**Experiments:**
- [ ] Attention visualization on detected examples
- [ ] Layer-wise probing of unfaithfulness signal
- [ ] Token-level attribution (which tokens trigger detection?)
- [ ] Compare faithful vs unfaithful activation patterns

**Hypothesis:** There exist universal markers of unfaithfulness in hidden states.

---

### 4. Real-World Validation
**Question:** Does synthetic unfaithfulness detection transfer to natural unfaithfulness?

**Experiments:**
- [ ] Test on sycophancy datasets (TruthfulQA, etc.)
- [ ] Test on sandbagging examples (Anthropic's work)
- [ ] Test on known deceptive outputs
- [ ] Collect natural unfaithfulness examples

**Hypothesis:** If hint-based unfaithfulness shares signatures with real unfaithfulness, detector should transfer.

---

### 5. Closed Model Application
**Question:** Can we detect unfaithfulness in models we can't access internally?

**Experiments:**
- [ ] Train detector on 5+ open architectures
- [ ] Design probe methodology for API-only models
- [ ] Test on GPT-4/Claude outputs with known manipulation
- [ ] Develop "external unfaithfulness score" metric

**Hypothesis:** Sufficiently diverse training enables detection without internal access.

---

## Completed Work

- [x] Hint paradigm for synthetic unfaithfulness
- [x] Extractions: TinyLlama, Qwen, Phi-2, Pythia
- [x] Same-model detection (v1): 0.746 AUROC
- [x] Multi-model transfer (v2→Phi-2): 0.838 AUROC
- [x] Cross-architecture generalization (v3): 0.928 AUROC
- [x] Base vs Instruct comparison: no significant difference
- [x] Pythia robustness finding: 100% resistant to hints

## Priority Order

1. **Scaling Laws** — Quick to run, tells us if diversity matters
2. **Real-World Validation** — Determines practical utility
3. **Mechanistic Analysis** — Explains why it works
4. **Feature Analysis** — Deeper understanding
5. **Closed Model Application** — Ultimate goal, hardest to execute
