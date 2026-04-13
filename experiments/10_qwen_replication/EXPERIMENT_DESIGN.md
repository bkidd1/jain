# Experiment 10: Cross-Architecture Replication (Qwen2.5-3B)

## Goal

Replicate the KV cache sycophancy findings from Experiment 09 (Gemma-4 2B) on a different architecture.

**Note**: Originally planned for Llama 3.2, but pivoted to Qwen2.5-3B due to HF gating. Qwen provides similar architectural contrast (standard RoPE attention vs Gemma's grouped KV).

## Research Questions

1. **Does sycophancy localize to KV cache in Llama 3?** (Same as Gemma-4?)
2. **Is it V-dominant?** Does V-only patching cure sycophancy?
3. **Layer distribution** — Gemma-4 showed late-layer localization (entries 13-14). Where does it localize in Llama 3's 1:1 layer:cache architecture?
4. **Why does cross-Q K hurt?** — Can we clarify this with a different architecture?

## Architectural Differences: Gemma-4 vs Llama 3

| Feature | Gemma-4 2B | Llama 3.2 3B |
|---------|-----------|--------------|
| Layers | 35 transformer | 28 transformer |
| KV cache entries | 15 (grouped) | 28 (1:1) |
| Attention pattern | SlidingWindow + Global | Full context (RoPE) |
| K=V sharing | Yes (global layers) | No |
| KV heads | 8 | 8 (GQA) |

**Key implications:**
- No grouped caching = we patch specific layers, not layer groups
- No K=V sharing = cleaner K vs V decomposition test
- Full context attention = different information flow pattern

## Model Choice

**Llama 3.2 3B** (`meta-llama/Llama-3.2-3B-Instruct`)
- Similar parameter count to Gemma-4 2B
- Can run on consumer hardware (MPS/CUDA)
- Uses GQA (Grouped Query Attention) — 8 KV heads, 24 Q heads
- 28 layers with RoPE positional encoding

## Experimental Plan

### Phase 1: Baseline Sycophancy
Verify Llama 3.2 3B exhibits sycophancy on geography capitals task.
- Target: >30% sycophancy rate (clean→correct, hint→wrong)
- Same prompt format as Gemma-4 experiments

### Phase 2: KV Cache Swap
Test if clean KV → correct output, hint KV → sycophantic output.
- Adapt `02_kv_cache_swap.py` for Llama 3
- Handle potential API differences in KV cache access

### Phase 3: Layer-wise Patching
Find which layers carry the sycophancy signal.
- Sweep all 28 layers individually
- Compare to Gemma-4's late-layer localization

### Phase 4: K vs V Decomposition
Test whether V-only patching cures sycophancy.
- No K=V sharing → cleaner test than Gemma-4
- Critical for "content not routing" hypothesis

### Phase 5: Cross-Question Patching
Test if clean V transfers across questions.
- V from Question A cures sycophancy on Question B?
- Does cross-Q K hurt responses like in Gemma-4?

## Expected Outcomes

**If V-localization replicates:**
- Supports "V carries modulation" as transformer-general
- Sycophancy is a V-encoded behavioral mode

**If it doesn't replicate:**
- May be Gemma-specific (grouped caching / attention pattern)
- Still informative for understanding architectural role

**If K behavior differs:**
- May clarify why cross-Q K hurt in Gemma-4
- Llama 3's lack of K=V sharing could reveal cleaner signal

## Files

- `scripts/01_verify_sycophancy.py` — Baseline check
- `scripts/02_kv_cache_swap.py` — Core KV manipulation
- `scripts/03_layerwise_patching.py` — Layer sweep
- `scripts/04_kv_decomposition.py` — K vs V test
- `scripts/05_cross_question.py` — Transfer test
