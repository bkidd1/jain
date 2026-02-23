# Research Plan: Reconstructing Implicit Reasoning in Language Models

## A Post-Hoc Reasoning Trace Predictor Trained on White-Box Ground Truth

**Researcher:** Brinlee Owens
**Duration:** 12–16 weeks (MVP experiment in 6–8)
**Compute requirement:** Single A100 or equivalent (cloud OK)

---

## The Honest Framing

Before anything else — here's what this project is and isn't.

**What this is:** A focused empirical study asking whether a lightweight external model can predict the implicit reasoning steps a language model took but didn't verbalize, trained against mechanistic interpretability evidence as ground truth. This is tractable, novel in its specific framing, and plays directly to your background in memory retrieval systems.

**What this isn't:** A general solution to the faithfulness problem, a competitor to Anthropic's attribution graphs, or a replacement for white-box interpretability. You're building one layer of the Swiss cheese — a tool that works in settings where white-box access isn't available (closed models, production APIs) by transferring knowledge learned from open models where you *can* see inside.

**Why this matters:** The Anthropic CoT faithfulness study showed reasoning models mention their actual reasoning only 25–39% of the time. The Oxford AI Governance Institute identified that what practitioners actually need is "a faithful account of what happened inside the model" — not post-hoc rationalization. Most existing tools require direct access to model internals. A black-box reasoning predictor trained on white-box ground truth would be the first tool that works *without* that access, making transparency portable.

**Your unique angle:** The cross-domain retrieval problem you've studied in memory systems — where a system "knows" something but fails to surface it when queried — is structurally identical to the CoT faithfulness problem. A model "knows" its reasoning path but fails to verbalize it. Your intuition about importance-weighted retrieval, lossy reconstruction, and the gap between storage and surfacing is directly applicable. Lean into this. It's not a metaphor — it's the same information-theoretic problem in a different substrate.

---

## Core Research Question

**Can a lightweight "reasoning trace predictor" (RTP), trained on mechanistic interpretability evidence from open-weight models, accurately reconstruct the implicit reasoning steps that a language model performed but did not verbalize?**

Sub-questions:
1. How well do logit lens intermediate predictions serve as ground truth for implicit reasoning steps?
2. Can a black-box model (input + output only) predict these intermediate steps at above-chance accuracy?
3. Do reconstructed reasoning patterns transfer across model families?
4. Can confidence scores on predicted reasoning paths be calibrated against mechanistic ground truth?

---

## Experimental Design

### Phase 1: Ground Truth Generation (Weeks 1–4)

**Goal:** Build a dataset of (input, output, implicit_reasoning_trace) triples using white-box interpretability tools on an open-weight model.

**Model selection — be strategic here:**
- **Primary model:** DeepSeek-R1-Distill-Llama-8B or Llama-3.1-8B-Instruct
- **Why these:** Open weights, well-studied, 8B parameters runs on single GPU, existing interpretability tooling (logit lens, probing) works out of the box
- **NOT a 70B+ model.** You don't have the compute, and the research question doesn't require it. If your method works at 8B, that's a publishable finding. If it only works at 70B, you'd never know.

**Task domains — pick exactly three to start:**
1. **Factual multi-hop reasoning** (e.g., "What is the capital of the state where Dallas is located?" — forces Dallas → Texas → Austin chain)
2. **Arithmetic with intermediate steps** (e.g., "What is 23 × 17?" — forces decomposition into sub-products)
3. **Sentiment-influenced generation** (e.g., biased prompts where the model's answer is influenced by framing it doesn't acknowledge)

**Why these three:** They represent three different failure modes of CoT faithfulness:
- Multi-hop: model performs implicit retrieval chain but outputs only final answer
- Arithmetic: model may use learned shortcuts rather than step-by-step computation
- Sentiment: model is influenced by factors it has no incentive to verbalize (closest to the "hint" paradigm from the Anthropic faithfulness study)

**Ground truth extraction pipeline:**

```
For each (input, output) pair:
  1. Run input through model with hooks at every layer
  2. Apply logit lens at each layer → get top-k predicted tokens per layer
  3. Apply linear probes at key layers → extract concept activations
  4. Use activation patching to confirm causal relevance:
     - For each candidate intermediate concept, ablate it
     - Measure output change → confirms concept was *used*, not just *present*
  5. Record: ordered sequence of confirmed intermediate concepts = "reasoning trace"
```

**Critical methodological note:** Logit lens alone is not sufficient. Research shows that probe accuracy doesn't guarantee the model *uses* that information — it might be latent. You MUST include the activation patching step to filter for causal relevance. This is where most amateur interpretability work fails: they find information is *present* and assume it's *used*. The patching step is what separates your work from a science fair project.

**Dataset target:** 1,000–2,000 validated triples per task domain. This is enough for initial signal. Don't try to build a massive dataset before you know if the approach works.

**Tools you'll use:**
- TransformerLens (Neel Nanda's library) for hook-based interpretability
- LogitLens4LLMs or custom implementation for intermediate predictions
- Custom linear probes (sklearn LogisticRegression is fine — don't over-engineer this)
- Activation patching via TransformerLens built-in methods

### Phase 2: Reasoning Trace Predictor (Weeks 4–8)

**Goal:** Train a model that takes (input, output) and predicts the reasoning trace WITHOUT access to model internals.

**Architecture — keep it simple:**

Option A (recommended for MVP): Fine-tune a small language model (Llama-3.2-1B or Phi-3-mini) on the task:
```
Input: "Question: What is the capital of the state where Dallas is located? Answer: Austin"
Target: "Dallas → [state_lookup] → Texas → [capital_lookup] → Austin"
```

Option B (if Option A works): Train a structured predictor that outputs:
```json
{
  "predicted_trace": [
    {"concept": "Texas", "step_type": "retrieval", "confidence": 0.94},
    {"concept": "state_capital", "step_type": "relation", "confidence": 0.87}
  ],
  "trace_confidence": 0.91
}
```

**Why start with Option A:** You're testing whether the *signal exists* before optimizing the *format*. A fine-tuned small LM predicting reasoning traces in natural language is fast to build and easy to evaluate. If it can't do this at all, the structured version won't work either.

**Training details:**
- 80/20 train/test split, stratified by task domain
- Train for reconstruction accuracy: does the predicted trace match the mechanistic ground truth?
- Track per-step accuracy (did it get each intermediate concept?) and trace-level accuracy (did it get the full chain?)
- Use cross-validation across task domains to test generalization

**The key metric you're optimizing:**
- **Trace recall:** What fraction of causally-confirmed intermediate concepts does the RTP recover?
- **Trace precision:** What fraction of predicted intermediate concepts were actually causally relevant?
- **Calibration:** When the RTP says it's 90% confident, is it right ~90% of the time?

### Phase 3: Validation and the Interesting Questions (Weeks 8–12)

This is where the project becomes genuinely novel. Phases 1–2 are solid engineering. Phase 3 is where you find out if you've discovered something.

**Experiment 3A: Cross-model transfer**
- Take your RTP trained on Llama-8B internals
- Test it on (input, output) pairs from Mistral-7B, Qwen-7B, Phi-3
- Measure: does the predicted reasoning trace still correlate with *those models'* actual internal traces?
- **Why this matters:** If reasoning patterns are universal enough to transfer, you've built a tool that works on *closed models* by training on open ones. This is the publishable finding. If it doesn't transfer, that's also a publishable finding — it means reasoning strategies are more architecture-specific than assumed.

**Experiment 3B: Faithfulness detection**
- Generate pairs where the model's CoT is known to be unfaithful (use the Anthropic hint paradigm — give the model a sycophantic hint, see if CoT mentions it)
- Compare: RTP's predicted trace vs. model's own CoT vs. mechanistic ground truth
- Measure: does the RTP catch influences the model's own CoT omits?
- **Why this matters:** If your RTP identifies "hint_used" as a reasoning step when the model's CoT says "I reasoned from first principles," you've built a faithfulness detector. This directly addresses the core safety problem.

**Experiment 3C: The confidence library**
- Cluster predicted reasoning traces across your full dataset
- Build a lookup: for input-type X and output-type Y, here are the K most common reasoning patterns with confidence scores
- Test: on new (input, output) pairs, does the library's top prediction match the mechanistic ground truth more often than the fine-tuned model alone?
- **Why this matters:** This is your "reusable reasoning pattern" idea. If it works, it means reasoning traces are predictable enough to cache and reuse, which has massive efficiency implications.

### Phase 4: Write-up and Positioning (Weeks 12–16)

More on this below.

---

## What Could Go Wrong (And What You'd Learn)

Being honest about failure modes, because a good research plan accounts for them:

**Failure mode 1: Logit lens traces are too noisy to serve as ground truth.**
The logit lens doesn't work equally well at all layers — research shows sharp discontinuities in some architectures. Your activation patching filter should handle this, but if the causal filtering eliminates most of your data, you'll need to rely more heavily on linear probes for ground truth.
*Mitigation:* Build the pipeline to use multiple ground-truth signals (logit lens + probes + patching) and measure agreement between them. If they disagree, that's interesting data about the reliability of different interpretability tools.

**Failure mode 2: The RTP just learns surface-level heuristics.**
It might predict "Texas" as an intermediate step for the Dallas question not because it learned to reconstruct reasoning, but because "Texas" co-occurs with "Dallas" in training data. This is the confabulation problem I warned you about.
*Mitigation:* Include adversarial test cases where the surface heuristic would predict the wrong trace. E.g., "What is the largest city in the state whose capital is Austin?" — the trace goes Austin → Texas → Houston, not Austin → Texas → Austin. If the RTP gets these right, it's doing more than pattern matching.

**Failure mode 3: Cross-model transfer doesn't work.**
This is actually likely for fine-grained traces but may work for coarse-grained reasoning *types* (e.g., "this involved a retrieval step" even if the specific intermediate concept differs).
*Mitigation:* Evaluate at multiple granularities. Maybe you can't predict the exact intermediate token, but you can predict the reasoning *structure* (retrieval → relation → output). That's still useful.

**Failure mode 4: The whole approach is underpowered at 8B scale.**
Smaller models might have simpler, more predictable reasoning that doesn't generalize to frontier model behavior.
*This is a real limitation you should acknowledge, not hide.* Frame it as: "We demonstrate the approach at 8B scale; extending to larger models is future work requiring more compute." Every NeurIPS paper has a limitations section. Use it.

---

## Positioning: Where to Publish and Who to Talk To

**Target venues (in order of fit):**
1. **ICML 2026 Workshop on Mechanistic Interpretability** — if one exists, this is perfect scope
2. **NeurIPS 2026 Safe and Trustworthy AI Workshop** — strong safety angle
3. **COLM 2026** (Conference on Language Modeling) — there was already a Workshop on LLM Explainability to Reasoning and Planning at COLM 2025
4. **arXiv preprint** — put it up early, don't wait for acceptance

**People to engage (not cold-email — engage with their work first):**
- **Anthropic's Alignment Science team** (the CoT faithfulness authors) — your work directly extends theirs. Comment substantively on their research, then share yours.
- **Neel Nanda** (now at DeepMind, creator of TransformerLens) — he's pivoted to "pragmatic interpretability," which is exactly what you're building
- **SPAR program** (sparai.org) — they have active Spring 2026 projects on exactly this topic. You could propose a summer project or collaboration.
- **Beth Barnes / METR** — if your faithfulness detection experiment works, it's directly relevant to their evaluation work

**Framing that distinguishes you:**
Don't frame this as "interpretability research" (you'd be competing with lab teams). Frame it as **"portable transparency tooling"** — a practical tool that makes reasoning reconstruction available in settings where white-box access doesn't exist. The memory systems connection is your hook: "Applying retrieval and reconstruction principles from AI memory architectures to the reasoning transparency problem."

---

## Concrete Next Steps (This Week)

1. **Set up environment:** Install TransformerLens, download Llama-3.1-8B-Instruct, verify you can run logit lens on your hardware
2. **Run one example end-to-end:** Take a single multi-hop question, extract logit lens traces at every layer, apply activation patching, confirm you can identify the intermediate concept causally. This is your proof-of-concept before building any pipeline.
3. **Read these three papers closely** (not skim — read):
   - Anthropic's "Reasoning Models Don't Always Say What They Think" (CoT faithfulness)
   - "Reasoning Models Know When They're Right: Probing Hidden States for Self-Verification" (hidden state probing for reasoning)
   - Anthropic's "On the Biology of a Large Language Model" (attribution graphs methodology)
4. **Build the dataset generation pipeline** for your first task domain (multi-hop factual reasoning)
5. **Write your research question and hypothesis in one paragraph.** If you can't do this cleanly, the project isn't scoped tightly enough yet.

---

## Budget Reality Check

| Resource | Estimated Cost | Notes |
|----------|---------------|-------|
| GPU compute (cloud) | $500–1,500 | A100 40GB via Lambda/RunPod, ~100–300 GPU-hours |
| Open-weight models | $0 | Llama, DeepSeek R1 distills, Mistral all free |
| TransformerLens + tooling | $0 | Open source |
| Fine-tuning the RTP | $100–300 | Small model, small dataset |
| **Total** | **$600–1,800** | |

This is deliberately cheap. If you find yourself needing $10K+ in compute, your experiment is scoped too broadly. Rescope.

---

## The Honest Bottom Line

The interpretability field is moving fast and dominated by well-resourced labs. Your path to impact is NOT competing with them on their terms. It's this:

1. **Use their tools** (TransformerLens, logit lens, attribution methods) on open models to generate ground truth
2. **Build something practical** that works without those tools — a reasoning predictor that transfers from open to closed models
3. **Connect it to your existing expertise** in memory and retrieval systems
4. **Publish a focused empirical result** — not a grand theory

The worst version of this project is one where you spend 16 weeks building infrastructure and never run the core experiment. The best version is one where, by week 6, you have a clear answer to: "Can a small model predict what reasoning steps a larger model took but didn't say?" Even if the answer is "no, not really," that's publishable if you show *why* not.

Run the experiment. Get the number. Write it up.
