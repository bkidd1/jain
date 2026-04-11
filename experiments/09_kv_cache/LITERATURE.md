# V vectors encode retrieval modes: what existing literature reveals

The experimental findings — that V vectors at late KV cache entries encode a transferable "factual retrieval mode," that this mode transfers across knowledge domains but not to computation, and that the signal requires full representational coherence — are strongly supported by converging evidence from at least six distinct research streams in mechanistic interpretability.

The literature provides a near-complete mechanistic framework: MLPs store factual knowledge as key-value memories, late-layer attention heads extract specific attributes via their OV circuits (the V vector pathway), and sycophancy operates by contaminating this extraction process with social-pressure signals that redirect retrieval away from truthful outputs. The sharp boundary at computation reflects a categorical architectural distinction: facts are looked up while arithmetic is computed through fundamentally different distributed circuits.

---

## The OV circuit literature establishes V vectors as retrieval functions

The foundational framework comes from Elhage et al.'s "A Mathematical Framework for Transformer Circuits" (2021), which decomposes each attention head into two independent computations: the QK circuit (determining where to attend) and the OV circuit (determining what information to move). The OV matrix W_V · W_O maps source tokens through a value-output transformation that writes linearly into the residual stream. This architecture means V vectors — combined with the output projection — carry the semantic content that gets transferred between positions. **K vectors route attention; V vectors carry payload.**

The most direct evidence for V vectors encoding factual retrieval comes from Geva et al.'s "Dissecting Recall of Factual Associations in Auto-Regressive Language Models" (EMNLP 2023), which reveals a three-step internal mechanism for factual recall:

1. First, early MLP sublayers enrich the subject token's representation with multiple attributes ("subject enrichment")
2. Second, relation information propagates via attention to the prediction position
3. Third — and crucially — upper-layer attention heads perform the actual attribute extraction, with subject-attribute mappings encoded in their OV parameters

This means the OV circuit (and hence V vectors) doesn't just passively relay information; it actively implements retrieval functions that extract specific factual attributes from enriched representations.

The "Fact Finding" circuit analysis on LessWrong provides granular confirmation: in an athlete-sport recall task, the attention pattern of the key extraction head doesn't change between athletes, meaning all factual differentiation (basketball vs. football vs. baseball) must be encoded in the V vectors' interaction with the attended representation.

Millidge and Black's SVD analysis of OV matrices (2022) shows that the principal computational directions of each attention head correspond to semantically coherent groupings — deleting specific singular vectors produces large effects on semantically related logits and small effects on unrelated ones. The MAPS framework (ACL 2025) extends this by showing that the expanded OV matrix E · W_V · W_O · E^T reveals specific knowledge relations (capital-of, language-of) that individual heads implement.

These findings collectively establish that **V vectors are not generic information conduits but encode structured, interpretable retrieval functions** — precisely what the experimental observation of a "factual world-knowledge retrieval mode" describes.

The relationship to MLP-stored knowledge is complementary rather than competing. The ROME paper (Meng et al., NeurIPS 2022) localizes factual storage to mid-layer MLP modules via causal tracing, treating each MLP as a key-value store where keys encode subjects and values encode knowledge. But ROME also identifies a critical late attention site where information is assembled for output. Editing this late attention site produces mere "regurgitation" rather than genuine knowledge change — establishing that attention V vectors implement a retrieval/routing mode rather than serving as the primary storage site.

Wu et al.'s "Retrieval Head" work (ICLR 2025) found that a sparse set (~5%) of attention heads are responsible for retrieving relevant information, and masking them causes the model to hallucinate while maintaining fluency. V vectors in these heads carry the retrieved factual content.

---

## Why the boundary falls exactly at computation

The sharp transfer boundary — 38–58% cure rate for all factual domains, exactly 0% for arithmetic — reflects what the literature reveals as a **categorical architectural distinction** between how transformers process retrieval versus computation.

### Factual retrieval operates through a localized lookup pipeline

ROME demonstrates that individual facts can be changed by rank-one edits to single MLP weight matrices, confirming that facts are discretely stored as key-value associations. The three-step Geva et al. pipeline (enrichment → relation propagation → extraction) is fundamentally a structured memory access operation. V vector patching restores the correct readout from this memory system.

### Arithmetic uses entirely different computational substrates

Nanda et al.'s "Progress Measures for Grokking via Mechanistic Interpretability" (ICLR 2023) fully reverse-engineered how a transformer performs modular addition: the model uses discrete Fourier transforms and trigonometric identities to convert addition into rotation operations — a learned algorithmic procedure, not a lookup.

Hanna et al.'s analysis of greater-than computation in GPT-2 (NeurIPS 2023) found the comparison mechanism distributed across multiple neurons with no single "greater-than neuron."

Most relevantly, Nikankin et al.'s "Arithmetic Without Algorithms" (ICLR 2025) showed that production LLMs perform arithmetic using neither robust algorithms nor pure memorization, but a "bag of heuristics" — an ensemble of many sparse MLP neurons, each implementing simple pattern-matching rules, whose collective activation produces the answer.

Stolfo et al. (EMNLP 2023) provide the most direct comparison, applying causal mediation analysis to both arithmetic and factual queries and finding that their information flow patterns are mechanistically distinct.

A 2025 paper ("Disentangling Recall and Reasoning in Transformer Models") provides causal evidence that recall and reasoning rely on separable but interacting circuits: disabling recall circuits reduces fact-retrieval accuracy by up to 15% while leaving reasoning intact, and vice versa. Neurons show near-binary task-specific firing patterns.

**The boundary is sharp because V vector patching intervenes on the retrieval pipeline without touching the computation pipeline.** There is no single stored "value" to restore for 7 × 8 = 56; the answer emerges from the collective firing of an ensemble of heuristic neurons processing the operands. Patching a "factual retrieval mode" into this circuit is mechanistically inert — it's restoring a memory readout function in a context that requires algorithmic execution.

---

## Late KV cache entries mediate the critical information relay

The localization to KV cache entry 13 of 15 (covering the last ~10 layers of a 35-layer model) aligns precisely with what the literature identifies as the **information routing phase** of factual recall.

ROME's causal tracing reveals a two-site mechanism:
- Mid-layer MLPs (~35–55% depth) store and extract factual associations at the subject token position
- Late-layer attention mechanisms (~70–95% depth) relay this information to the final prediction position

The KV cache at these late layers stores exactly the keys and values these routing attention heads need. Corrupting entry 13/15 disrupts the attention-mediated information transfer that completes factual recall.

The logit lens (nostalgebraist, 2020) and tuned lens (Belrose et al., 2023) confirm that factual predictions crystallize through iterative refinement that converges in late layers. Early layers produce shallow guesses; the exact entity-level prediction emerges only in the final layers. The logit lens reveals that inputs are immediately converted to a different representation space and then smoothly refined toward the final prediction — with late layers performing what Geva et al. (EMNLP 2022) characterize as "saturation" (boosting the correct answer) and "elimination" (suppressing incorrect candidates).

The NLP-pipeline view provides additional support. Tenney et al. (ACL 2019) showed BERT implements the classical NLP pipeline in order: POS → parsing → NER → semantic roles → coreference, with semantic tasks concentrated in upper layers. Jawahar et al. (ACL 2019) found the same hierarchy: surface features in lower layers, syntactic features in middle layers, semantic features in upper layers. Factual retrieval is a high-level semantic operation requiring the integration of entity knowledge, relational understanding, and output selection — all functions the literature places in late layers.

---

## Transformers demonstrably encode distinct processing modes

The concept of V vectors encoding a transferable "processing mode" has direct precedent in three convergent research programs.

### Function Vectors

Todd et al.'s "Function Vectors in Large Language Models" (ICLR 2024) discovered that transformers represent input-output functions as compact vectors during in-context learning. A small number of attention heads transport these function vectors, which trigger task execution even in zero-shot settings entirely unlike the ICL contexts from which they were extracted.

Function vectors can be composed: summing the vectors for "provide the capital city" and "last item in list" produces a vector that correctly executes the combined function. Critically, function vectors contain information encoding the output space of the function plus additional procedural information about how to process — not just what to output.

### Task Vectors

Hendel et al.'s "In-Context Learning Creates Task Vectors" (EMNLP 2023 Findings) independently discovered that ICL compresses demonstration sets into single task vectors, with t-SNE visualization revealing tight clustering by task type. Similar tasks cluster closer together, retaining ~80–90% of full ICL accuracy.

This clustering structure directly predicts the observed transfer pattern: a factual retrieval task vector should cluster with all knowledge-retrieval tasks (geography, literature, history, science) but be distant from computation tasks (math).

### Representation Engineering

The representation engineering framework (Zou et al., 2023) generalizes this, demonstrating that high-level cognitive properties — honesty, harmlessness, power-seeking — are encoded as nearly linear directions in activation space, readable via probing and controllable via activation addition.

Turner et al.'s activation addition work (AAAI 2024) and Li et al.'s inference-time intervention (NeurIPS 2023) confirm that abstract processing properties can be captured as directions in attention head activations and causally used to steer model behavior.

Park et al. (ICML 2024) formalized this as the **Linear Representation Hypothesis**, proving that causally separable concepts are represented as orthogonal vectors under a causal inner product — meaning "factual retrieval" and "computation" modes, if causally separable, would occupy orthogonal subspaces, explaining non-transfer.

---

## Key References

- Elhage et al. (2021) - "A Mathematical Framework for Transformer Circuits"
- Geva et al. (EMNLP 2023) - "Dissecting Recall of Factual Associations"
- Meng et al. (NeurIPS 2022) - ROME
- Wu et al. (ICLR 2025) - "Retrieval Head"
- Nanda et al. (ICLR 2023) - "Progress Measures for Grokking"
- Nikankin et al. (ICLR 2025) - "Arithmetic Without Algorithms"
- Stolfo et al. (EMNLP 2023) - Causal mediation analysis
- Todd et al. (ICLR 2024) - "Function Vectors in LLMs"
- Hendel et al. (EMNLP 2023) - "ICL Creates Task Vectors"
- Zou et al. (2023) - Representation Engineering
- Park et al. (ICML 2024) - Linear Representation Hypothesis
- Belrose et al. (2023) - Tuned Lens
- nostalgebraist (2020) - Logit Lens
