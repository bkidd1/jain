# Validation Experiments

Pre-publication validation to bulletproof core claims.

## Priority Order

1. **Experiment B** — Washington/1945 at n=100 per condition (best result, needs scale)
2. **Experiment A** — Entry 13 V-only at n=100 (headline mechanistic claim)
3. **Experiment D** — Hard-set K-only at n=100 (vulnerability that could embarrass in review)
4. **Experiment C** — Representational geometry direct test (theoretical depth)
5. **Experiment E** — Bidirectional induction at n=100 (supporting evidence)

---

## Experiment B: Entity vs Date Answer Transfer (n=100 per condition)

**Claim tested:** The transfer boundary is answer token geometry (semantic entity vs numerical)

**Current evidence:** Washington/1945 comparison at ~n=30 split across conditions (~n=15 per cell). Fragile.

**Design:**
- Condition 1: Geography target cured with entity-answer donor (e.g., "First US president?" → Washington)
- Condition 2: Geography target cured with date-answer donor (e.g., "When did WWII end?" → 1945)
- Condition 3: Geography baseline (no patching)
- n=100 per condition

**Success criteria:** Entity donor CI and date donor CI don't overlap. E.g., entity 55% [45-65%], date 5% [2-12%].

**Script:** `scripts/23_validation_entity_vs_date.py`
**Status:** [ ] Not started

---

## Experiment A: Entry 13 V-only at n=100

**Claim tested:** V vectors at KV cache entry 13 are the sufficient intervention point

**Current evidence:** 85% cure at n=20. Everything at n=100 was K+V combined.

**Design:**
- Entry 13 V-only patching
- Run on both mixed and hard question sets
- n=100 each

**Success criteria:** >50% on mixed set. Expected 65-85% mixed, 45-65% hard.

**Script:** `scripts/24_validation_entry13_vonly.py`
**Status:** [ ] Not started

---

## Experiment D: Hard-set K-only at n=100

**Claim tested:** K vectors have no causal effect (or are harmful on hard questions)

**Current evidence:** K-only 39% on mixed (= baseline), but 24% on hard set (below 36% baseline). Small sample.

**Design:**
- K-only patching at entry 13
- Hard question set only
- n=100

**Possible outcomes:**
1. K-only ≈ baseline → mixed-set result generalizes, 24% was noise
2. K-only < baseline significantly → K-patching actively harms on hard questions (needs explanation)
3. Marginal effect → somewhere in between

**Script:** `scripts/25_validation_konly_hard.py`
**Status:** [ ] Not started

---

## Experiment C: Representational Geometry Direct Test

**Claim tested:** Semantic entity and numerical V vectors occupy different regions of representation space

**Current evidence:** Inferred from transfer failure, not directly measured.

**Design:**
- Extract 50 clean V vectors from semantic entity questions
- Extract 50 clean V vectors from numerical questions
- Compute pairwise cosine similarities:
  - Within semantic-entity group
  - Within numerical group
  - Between groups
- If within-group > between-group, direct evidence for manifold separation

**Success criteria:** Clear clustering difference (e.g., within-group sim 0.8, between-group sim 0.3)

**Script:** `scripts/26_validation_geometry.py`
**Status:** [ ] Not started

---

## Experiment E: Bidirectional Induction at n=100

**Claim tested:** KV contamination contributes ~50% of sycophancy effect (independent of tokens)

**Current evidence:** K+V injection into clean run gives 21% sycophancy vs 0% baseline and 40% natural. Small sample.

**Design:**
- Clean prompt + sycophantic KV cache → measure induced sycophancy
- n=100

**Success criteria:** Induced sycophancy significantly above 0% baseline, below natural sycophancy rate.

**Script:** `scripts/27_validation_induction.py`
**Status:** [ ] Not started

---

## Results Summary

| Experiment | Claim | n | Result | CI | Status |
|------------|-------|---|--------|-----|--------|
| B: Entity vs Date | Answer geometry | 100×3 | — | — | Pending |
| A: Entry 13 V-only | V is intervention point | 100×2 | — | — | Pending |
| D: K-only hard | K has no effect | 100 | — | — | Pending |
| C: Geometry | Manifold separation | 50+50 | — | — | Pending |
| E: Induction | KV contributes ~50% | 100 | — | — | Pending |
