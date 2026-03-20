# Research Roadmap: Divergence Detection

## Current Status (Updated 2026-03-20)

**Key Finding:** Cross-architecture training outperforms same-model training for unfaithfulness detection.
- Detector trained on Qwen+Phi-2 achieves 0.928 AUROC on TinyLlama
- Detector trained on TinyLlama achieves only 0.746 AUROC on TinyLlama
- Delta: +18.2 percentage points
- 3-model detector (Qwen+Phi-2+TinyLlama-Base) achieves 0.943 AUROC

**Paper draft complete:** `paper/cross_architecture_detection.md`

**Field context:** Recent work (Chua & Evans 2025, Anthropic 2025, MATS 8.0) focuses on reasoning models. Our cross-architecture finding is novel but needs validation on reasoning models to maximize impact.

---

## Priority 1: Reasoning Model Testing (IN PROGRESS 🔥)

**Question:** Does cross-architecture transfer work on reasoning models?

**Why this matters:** The field has moved to reasoning models (DeepSeek R1, o1, etc.). Showing our approach works on these models would significantly increase impact.

**Experiments:**
- [ ] Extract from DeepSeek-R1-Distill-Qwen-7B ← **RUNNING NOW**
- [ ] Test existing 3-model detector on reasoning model extractions
- [ ] Train new detector including reasoning model data
- [ ] Compare: do reasoning models show different unfaithfulness patterns?

**Expected outcome:** Either (a) our detector transfers to reasoning models, or (b) reasoning models have different patterns requiring mixed training.

---

## Priority 2: Mechanistic Analysis

**Question:** Why does cross-architecture training work better than same-model?

**Why this matters:** Explaining the mechanism makes our finding more compelling and actionable. The MATS 8.0 "nudged reasoning" hypothesis gives us a testable framework.

**Experiments:**
- [ ] Probe detector internal representations (v1 vs v3)
- [ ] Layer-wise analysis: where does unfaithfulness signal emerge?
- [ ] Compare attention patterns between same-model and cross-arch detectors
- [ ] Test "nudged reasoning" hypothesis: does bias accumulate across layers?

**Hypothesis:** Same-model training overfits to architecture-specific artifacts. Cross-architecture training forces learning of general unfaithfulness signatures.

---

## Priority 3: Architecture Scaling (Partially Complete ✓)

**Question:** How does # of training architectures affect detection?

**Completed:**
| # Archs | Training Data | AUROC on TinyLlama |
|---------|---------------|-------------------|
| 1 | TinyLlama (same) | 0.746 |
| 1 | Qwen only | 0.702 |
| 1 | Phi-2 only | 0.564 |
| 2 | Qwen + Phi-2 | 0.928 |
| 3 | Qwen + Phi-2 + TinyLlama-Base | **0.943** |

**Key finding:** Phase transition at 2 architectures. 3 > 2 by +1.5pp.

**Remaining (lower priority):**
- [ ] 4+ models (Mistral, Llama-8B, etc.)
- [ ] Plot scaling curve
- [ ] Determine if there's a ceiling

---

## Priority 4: Real-World Validation (Partially Complete ✓)

**Completed:**
- [x] Sycophancy paradigm: 59.4% detection, 100% precision

**Remaining (lower priority):**
- [ ] TruthfulQA benchmark
- [ ] Sandbagging examples (Anthropic)
- [ ] Natural unfaithfulness collection

**Note:** Sycophancy result already demonstrates cross-paradigm transfer. Additional validation is nice-to-have but not blocking.

---

## Priority 5: Closed Model Application (Long-term Goal)

**Question:** Can we detect unfaithfulness in API-only models?

**Experiments:**
- [ ] Train on 5+ diverse open architectures
- [ ] Design external probe methodology
- [ ] Test on GPT-4/Claude with known manipulation
- [ ] "External unfaithfulness score" metric

**Blocked on:** More architecture diversity, probe methodology design

---

## Completed Work

### Extractions
- [x] TinyLlama-1.1B-Chat
- [x] TinyLlama-1.1B (base)
- [x] Qwen2-1.5B
- [x] Qwen2-1.5B-Instruct
- [x] Phi-2
- [x] Pythia-1.4B (0% unfaithfulness — robust to hints)
- [ ] DeepSeek-R1-Distill-Qwen-7B ← in progress

### Detectors
- [x] v1 (TinyLlama only): 0.746 AUROC same-model
- [x] v2 (TinyLlama + Qwen): 0.967 AUROC, 0.838 transfer to Phi-2
- [x] v3 (Qwen + Phi-2, exclude target): 0.928 AUROC on TinyLlama
- [x] 3-model (Qwen + Phi-2 + TinyLlama-Base): 0.943 AUROC

### Key Findings
- [x] Cross-architecture beats same-model (+18.2pp)
- [x] Phase transition at 2 architectures
- [x] 3 models > 2 models (+1.5pp)
- [x] Pythia-1.4B immune to hint manipulation
- [x] Sycophancy transfer: 59.4% recall, 100% precision
- [x] Base vs Instruct: no significant difference

### Documentation
- [x] RESULTS.md with all metrics
- [x] Paper draft: `paper/cross_architecture_detection.md`

---

## Next Actions

1. **Now:** Wait for DeepSeek-R1-Distill-Qwen-7B extraction to complete
2. **Next:** Test detector transfer to reasoning model
3. **Then:** Mechanistic analysis OR train mixed detector (depending on transfer results)
4. **Finally:** Update paper with reasoning model results
