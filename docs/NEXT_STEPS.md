# Next Steps for Jain

## ✅ Completed (Week 1)

1. **Environment setup** — TransformerLens, Llama 3.1 8B, Mistral 7B
2. **Basic logit lens experiments** — Confirmed intermediate layer predictions work
3. **Activation patching** — Verified concepts are *causally used* (Texas: 96% effect)
4. **Cross-model transfer test** — 4/5 prompts show same concepts in Llama & Mistral
5. **Related work analysis** — Identified gap and novel contribution

**Key result:** Transfer is plausible! Same reasoning patterns appear in different model architectures.

---

## 📋 Week 2: Build the Dataset

### Goal
Generate 1,000+ (input, output, reasoning_trace) triples with causal verification.

### Tasks

- [ ] **2.1 Scale up extraction pipeline**
  - Run `tuned_lens_extraction.py` on 500+ multi-hop prompts
  - Use dataset generators from `src/dataset.py`
  - Store results in `data/processed/traces_llama.jsonl`

- [ ] **2.2 Implement proper Tuned Lens** (optional improvement)
  - Current: raw logit lens
  - Better: Use `tuned-lens` library with pre-trained translators
  - Trade-off: More setup vs. cleaner ground truth

- [ ] **2.3 Quality filtering**
  - Only keep traces where causal_effect > 0.1
  - Verify trace makes semantic sense
  - Split: 80% train, 20% test

### Output
- `data/processed/traces_llama.jsonl` — Training data
- `data/processed/traces_llama_test.jsonl` — Held-out test set

---

## 📋 Week 3-4: Train the Reasoning Trace Predictor (RTP)

### Goal
Fine-tune a small LM to predict reasoning traces from (input, output) pairs.

### Tasks

- [ ] **3.1 Format training data**
  ```
  Input: "Question: {prompt} Answer: {output}"
  Target: "{concept1} → {concept2} → {output}"
  ```

- [ ] **3.2 Fine-tune predictor**
  - Base model: Llama-3.2-1B or Phi-3-mini (small, fast)
  - Method: Standard causal LM fine-tuning
  - Epochs: 3-5, early stopping on validation loss

- [ ] **3.3 Evaluate on held-out Llama data**
  - Metric: Trace recall (% of concepts recovered)
  - Metric: Trace precision (% of predicted concepts that were real)

### Output
- `models/rtp_v1/` — Trained predictor checkpoint
- `experiments/rtp_v1_eval.json` — Evaluation results

---

## 📋 Week 5-6: The Transfer Experiment (The Money Shot)

### Goal
Test if RTP trained on Llama traces can predict Mistral's reasoning.

### Tasks

- [ ] **4.1 Extract ground truth from Mistral**
  - Run same extraction on Mistral 7B
  - `data/processed/traces_mistral.jsonl`

- [ ] **4.2 Test RTP on Mistral**
  - Use RTP (trained on Llama) to predict Mistral traces
  - Compare predicted traces to actual Mistral ground truth

- [ ] **4.3 Analyze transfer**
  - Does it work? At what granularity?
  - Fine-grained (exact tokens): probably fails
  - Coarse-grained (reasoning structure): might work

### Output
- `experiments/transfer_results.json` — The headline finding

---

## 📋 Week 7-8: Write-up & Positioning

### Goal
Draft a workshop paper or arXiv preprint.

### Tasks

- [ ] **5.1 Write introduction + related work**
- [ ] **5.2 Write methodology section**
- [ ] **5.3 Write results + analysis**
- [ ] **5.4 Create figures**
  - Logit lens heatmaps
  - Transfer accuracy by prompt type
  - Causal effect distributions

### Target Venues
- ICML 2026 Mechanistic Interpretability Workshop
- NeurIPS 2026 Safe & Trustworthy AI Workshop
- arXiv preprint (can submit anytime)

---

## 🎯 Success Criteria

| Outcome | What it means |
|---------|---------------|
| Transfer works (>50% overlap) | 🎉 Novel contribution, strong paper |
| Transfer partially works (20-50%) | 📝 Interesting finding, workshop paper |
| Transfer fails (<20%) | 📊 Negative result, still publishable if well-analyzed |

**Any of these is a valid research outcome.** The goal is to run the experiment rigorously and report what we find.

---

## 🚀 Immediate Next Action

**Start Week 2: Scale up the dataset.**

```bash
cd /Users/demosthenes/jain
source .venv/bin/activate
python -c "from src.dataset import generate_full_dataset; generate_full_dataset()"
```

Then run extraction on the generated prompts.

---

*Last updated: 2026-02-23*
