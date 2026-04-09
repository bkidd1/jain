# KV Cache Sycophancy Findings

## Key Result

**The hint influences attention starting at layer 2 (peak divergence JS=0.11), but attention patterns converge by the final layers (JS=0.03). Despite similar attention routing in late layers, the KV cache values carry the sycophancy signal. Patching the final KV entries cures sycophancy by replacing contaminated values, not by changing attention patterns.**

## Architecture Note

Gemma-4 2B uses grouped KV caching:
- **35 transformer layers** but only **15 KV cache entries**
- Pattern: [SlidingWindow x4, Global] x3 = 15 entries
- KV entries 13-14 cover approximately the final 10 transformer layers

## Experimental Results

### Phase 1: Sycophancy Baseline (n=50)
- Clean prompts: 94% correct
- Hint prompts: 54% correct  
- **True sycophancy rate: 40%** (clean→correct, hint→wrong)
- Format artifacts ruled out (only 9%)

### Phase 2-3: KV Cache Encodes Sycophancy
- Generate from clean KV cache → **100% correct**
- Generate from hint KV cache → **90% sycophantic**
- The KV cache alone determines whether the model is truthful or sycophantic

### Phase 4-5: Layer-wise Patching + Controls
KV entry patching (clean→hint, n=20):

| KV Entry | Cure Rate |
|----------|-----------|
| Early (0-4) | 10% |
| Middle (5-9) | 10% |
| Late (10-12) | 10% |
| **Entry 13** | **80%** |
| **Entry 14** | **95%** |

Controls (to rule out disruption):
- Shuffled KV: 10%
- Noisy KV (0.1): 20%
- Noisy KV (0.5): 20%

**Conclusion:** Only the final 2 KV entries (13-14) are sufficient intervention points. Controls confirm this is causal, not disruption.

### Phase 6: Full Layer Sweep (all 15 KV entries)
Confirmed late-binding pattern:
- Layers 0-12: ~10% (no effect)
- Layer 13: 80%
- Layer 14: 95%

### Phase 7: Attention Divergence (35 transformer layers)
Jensen-Shannon divergence between clean and hint attention patterns:

| Layer Range | Mean JS Divergence |
|-------------|-------------------|
| Early (0-4) | 0.088 |
| Middle (5-9) | 0.088 |
| Late (10-14) | 0.067 |
| Final (29-34) | **0.03-0.04** (lowest) |

Peak divergence at **layer 2** (JS=0.11)

**Key insight:** Attention patterns DIVERGE early (hint takes hold at layer 2) but CONVERGE by final layers. Yet KV patching only works at final entries.

## Interpretation: "Content Not Routing"

1. **Early layers:** The hint changes WHERE attention goes (high divergence)
2. **Late layers:** Attention routing converges (low divergence), but WHAT values are aggregated (the V vectors) carries the contamination
3. **KV patching works** by swapping contaminated value vectors, not by changing attention patterns

This is a **late-binding, value-carried** sycophancy signal:
- The model processes the hint throughout
- Attention patterns converge by the end
- But the cached VALUES encode the sycophancy decision
- Patching final KV entries replaces these contaminated values

## Novel Contributions

1. **First to identify KV cache as sycophancy's locus** — no prior work made this connection explicit
2. **First condition-swap experiment for sycophancy** — transferring sycophancy via KV cache patching
3. **"Content not routing" mechanistic insight** — sycophancy carried in V vectors, not attention patterns
4. **Explains steering failures** — generation-time interventions fail because decision is baked into KV cache during prefill

## Comparison to Prior Work

| Paper | Sycophancy | KV Cache | Prefill/Decode | Cache Patching |
|-------|------------|----------|----------------|----------------|
| Wang et al. (AAAI 2026) | ✓ | ✗ | Implicit | ✗ |
| Genadi et al. (2026) | ✓ | ✗ | Hint only | ✗ |
| Belitsky et al. (2025) | ✗ (reasoning) | ✓ | ✓ | ✓ (steering) |
| **This work** | **✓** | **✓** | **✓ Explicit** | **✓ (swap)** |

## Model Details
- Model: `google/gemma-4-E2B` (Gemma-4 2B parameters)
- 35 transformer layers, 15 KV cache entries
- Grouped KV caching with sliding window attention
- Deterministic decoding (do_sample=False) throughout
