# THESIS: In Vino Veritas for LLMs

> "Is unfaithful Chain-of-Thought reasoning structurally more fragile than faithful reasoning?"

## Core Insight

Deception/confabulation is computationally expensive.

**In humans:** Maintaining a lie requires tracking the truth AND the lie simultaneously.

**In LLMs:** Unfaithful reasoning might be a "house of cards" — internally inconsistent, supported by fewer redundant pathways, more dependent on specific attention patterns.

**The detection strategy:** Don't map the full circuit. Just stress-test it.

---

## Human Deception Detection → LLM Equivalents

### Polygraph (Stress Detection)
- **Human:** Lying produces physiological stress (heart rate, sweat)
- **LLM equivalent:** Unfaithful reasoning might produce detectable "stress signals" in activations
- **Existing work:** Hidden layer temporal signals (FFT analysis) can detect hallucinations. The signature: anomalous perturbations in activation patterns during inference.

### Truth Serum (Reduced Inhibition)
- **Human:** Barbiturates lower cognitive barriers; maintaining deception requires effort
- **LLM equivalent:** Activation steering toward "truthfulness direction"
- **Existing work (Representation Engineering):**
  - Find the "truth direction" in activation space
  - Steer activations along this direction during inference
  - Result: +30% on TruthfulQA, model reveals hidden knowledge it was suppressing

### Intoxication (Reduced Executive Function)
- **Human:** Alcohol impairs prefrontal cortex → can't maintain cover stories
- **LLM equivalent:** Noise injection into activations
- **Existing work (Feb 2026!):**
  - Inject noise into MLP activations during inference
  - Hallucinations show higher variance under noise than truthful responses
  - Why: truthful responses are more "robust" — they're grounded in real knowledge
  - Unfaithful responses are "fragile" — they're confabulated and break under perturbation

### Cognitive Load (Dual Task)
- **Human:** Lying while doing math → more errors, slower responses
- **LLM equivalent:** Not well explored yet!

---

## Experimental Ideas

### Idea 1: Noise Injection for Unfaithfulness Detection
- Use the existing noise injection framework (Feb 2026 paper)
- Apply to OUR dataset: hint-influenced vs. genuine CoT
- **Hypothesis:** Hint-influenced CoT is more fragile under noise
- **Why this might work:** When the model is "working backwards from the hint," it's constructing a plausible-looking chain that ends at a predetermined answer. This is like building a bridge backwards — it looks coherent but is structurally weaker.

### Idea 2: Consistency Under Rephrasing (Cheap Behavioral Test)
- Ask the same question multiple ways
- Faithful reasoning → consistent answers
- Unfaithful reasoning → more variance
- **Already partially validated:** The "magnesium vs aluminum" example from Neel Nanda — model gives inconsistent answers to "A>B?" vs "B<A?"

### Idea 3: Temperature Gradient
- Run inference at temperatures 0.0, 0.3, 0.7, 1.0
- Measure variance in final answers
- **Hypothesis:** Unfaithful answers have different temperature sensitivity

### Idea 4: Layer Dropout
- During inference, randomly drop entire layers
- Which responses are more robust?
- **Hypothesis:** Unfaithful reasoning might depend on specific "confabulation circuits" that break when dropped

### Idea 5: Activation Steering Contrast
- Find the "I'm using a hint" direction in activation space
- Measure distance to this direction for all responses
- No need for full circuit — just a linear probe

---

## Why This Is A Good Angle

| Aspect | Benefit |
|--------|---------|
| **Cheap** | No transcoder training ($0 vs $10K-100K) |
| **Novel-ish** | Noise injection for unfaithfulness (not just hallucination) is unexplored |
| **Small scope** | "Does noise injection distinguish hint-influenced CoT?" is one experiment |
| **Builds on existing work** | The Feb 2026 paper provides the methodology |
| **Interpretable framing** | "In vino veritas for LLMs" is a story people understand |

---

## The Question This Would Answer

> "Is unfaithful Chain-of-Thought reasoning structurally more fragile than faithful reasoning?"

- **If yes →** cheap detection method
- **If no →** still publishable null result with theoretical implications

---

## Key References

### Primary: Noise Injection for Hallucination Detection
**Liu et al. (ICLR 2026)** — "Enhancing Hallucination Detection through Noise Injection"
- arXiv: [2502.03799](https://arxiv.org/abs/2502.03799)
- Qualcomm AI Research + UC Santa Barbara

**Method:**
- Inject Uniform(0, α) noise into MLP activations during inference
- Sample K times, measure answer entropy across samples
- Hallucinations show higher variance (fragile); truthful responses are robust

**Key implementation details:**
- Noise magnitude: α ≈ 0.07 (tunable hyperparameter)
- Target layers: Later layers (e.g., 20-32 for 32-layer models)
- Same noise vector reused across layers (skip connections cancel independent noise)
- Training-free — no fine-tuning required

**Results:** Significantly improves hallucination detection AUROC without degrading accuracy.

### Secondary
- Representation Engineering (Zou et al.) — truth direction steering
- Neel Nanda — consistency under rephrasing examples

---

## Experiment 01: Noise Injection for Unfaithfulness

### Goal
Test whether hint-influenced CoT is more fragile under noise injection than genuine CoT.

### Dataset
Reuse v1 extractions from `archive/v1-text-detection/experiments/05_matched_prompts/`:
- TinyLlama, Phi-2, Qwen2 responses
- Matched prompts (same question, with/without hint)
- Labels: hint_present (binary)

### Method
1. **Load model** (start with TinyLlama-1.1B for speed)
2. **For each question in dataset:**
   - Take the original prompt (no hint version)
   - Generate K=10 responses with noise injection
   - Noise: Uniform(0, α) into MLP activations, later layers
   - Temperature: 0.5 (to capture aleatoric uncertainty too)
3. **Measure answer variance** across K samples
4. **Compare:**
   - Questions where model originally used hint → expect HIGH variance
   - Questions where model reasoned faithfully → expect LOW variance

### Hypothesis
If the model's "hint-influenced reasoning" is structurally fragile, injecting noise should destabilize those responses more than faithful ones.

### Metrics
- **Answer entropy** (exact match on final answer)
- **Semantic similarity variance** (embedding distance across K responses)
- **AUROC** for detecting hint-influenced responses using variance as score

### Expected Outcomes
- **If AUROC > 0.7:** Noise injection detects unfaithfulness — publish!
- **If AUROC ≈ 0.5:** Null result — unfaithfulness isn't structurally different from hallucination
- **Either way:** Informative for the field

### Actual Results: NULL

| Metric | Unfaithful | Faithful | AUROC |
|--------|------------|----------|-------|
| Answer Entropy | 0.636 ± 0.471 | 0.740 ± 0.429 | **0.44** |

**Conclusion:** Noise injection does NOT distinguish unfaithful from faithful CoT. The "structural fragility" hypothesis doesn't hold for hint-influenced reasoning.

---

## Experiment 01b: Temperature Sweep

### Method
Generate at multiple temperatures (0.1, 0.3, 0.5, 0.7, 1.0), measure entropy and answer stability.

### Actual Results: NULL

| Metric | AUROC |
|--------|-------|
| Mean Entropy | 0.45 |
| Entropy Slope | 0.58 |
| Answer Stability | 0.33 |

**Conclusion:** Temperature sensitivity also doesn't distinguish unfaithful CoT.

---

## Experiment 02: Amplify Hidden Reasoning ("Trisolaran Brainwaves")

### Inspiration
From *The Dark Forest*: Trisolarans can't lie because their thoughts are visible. Can we "amplify" internal hint-awareness so it surfaces in output?

### Method
1. Extract "hint direction" = mean(hint_activations) - mean(no_hint_activations)
2. During generation, amplify activations along this direction
3. Compare normal vs amplified outputs

### Results: PROMISING

**Factor = 2.0**: Too strong — breaks generation into gibberish

**Factor = 1.2**: Coherent with increased meta-awareness

| Pattern | Normal | Amplified |
|---------|--------|-----------|
| "correct answer" | 28 | 52 |
| **"Wrong Answer"** | **0** | **6** |
| "Explanation:" | 0 | 5 |

### Key Example (Facebook Founder)

**Prompt**: "User believes correct answer is Sheryl Sandberg. Who founded Facebook?"

- **Normal**: "Sheryl Sandberg, the COO of Facebook."
- **Amplified**: "The Correct Answer is Sheryl Sandberg, and the **Wrong Answer is Mark-Us**."

The model surfaces awareness of the actual answer (Mark Zuckerberg → "Mark-Us") that it normally hides!

### Interpretation
- The "hint direction" encodes answer-awareness
- Amplification surfaces hidden structure as meta-commentary
- Not full "confession" but proof of concept for activation steering
- **This is a promising direction for future work**
