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

- Feb 2026 noise injection paper (TODO: add citation)
- Representation Engineering / truth direction work
- Neel Nanda's consistency examples
