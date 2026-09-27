# SHARE-RAG / HBF-RAG current findings and risk assessment — 2026-09-27

## 1. Current decision point

The experiments now show that a pre-sense optimization opportunity can exist, but the useful operating point is conditional.

The key question for the collaboration is no longer simply whether pages can be safely pruned. The key question is:

> **If compressed-only retrieval already has more than 90% exact top-k recall, is it still necessary to preserve exact raw-vector reranking strongly enough to justify a pre-sense mechanism?**

This decision affects whether the current sweet spot is a meaningful systems target or only an exact-preservation corner case.

## 2. Analysis path

We decomposed the problem into three assumptions.

1. **A1 — raw reranking necessity.** After strong compression, does compressed-only retrieval still differ enough from exact raw-vector reranking to justify keeping a raw-vector path?
2. **A2 — safe page-pruning opportunity.** Given deterministic safe bounds, can candidate pruning translate into whole-page pruning?
3. **A3 — post-submit / pre-sense opportunity.** Of the pages that eventually become safely prunable, how many become prunable after host submission but before `sense_start`?

The safe candidate bound used in the current prototype is:

[
L_i = max(0,|q-hat{x}_i|-e_i), qquad
U_i = |q-hat{x}_i|+e_i
]

with pruning only when:

[
L_i > 	heta
]

where (	heta) is the current safe top-k threshold.

## 3. Why synthetic and real-data results differ

The first synthetic prototype showed substantial page-pruning opportunity for medium-length PQ codes. This did not transfer directly to real semantic embeddings.

The important quantity is not code length by itself, but the relation between reconstruction error and the top-k distance margin:

[
	ext{safe pruning difficulty}
propto
rac{	ext{quantization error}}
{	ext{candidate distance}-	heta^*}
]

Synthetic queries were generated near existing database vectors, creating relatively large distance margins. Real normalized semantic embeddings have much denser distance distributions, so the same relative quantization error can destroy the safe-pruning condition.

This explains the threshold-like behavior observed on real data: page pruning remains near zero until the reconstruction error becomes sufficiently small, then rises sharply.

## 4. 768-d real-data sweet spot

Using a 768-d embedding setup, 4 KiB pages hold about two FP16 raw vectors. The candidate budget is (R=512), with exact raw reranking used as the reference.

| Method | SciFact compressed R@10 | FiQA compressed R@10 | SciFact oracle page prune | FiQA oracle page prune | Ideal pre-sense sense reduction |
|---|---:|---:|---:|---:|---:|
| PQ48B | 70.5% | 74.1% | 0.0% | 0.0% | 0.0% / 0.0% |
| PQ96B | 82.8% | 83.0% | 0.0% | 0.0% | 0.0% / 0.0% |
| PQ192B | 90.5% | 90.2% | 6.3% | 0.5% | 0.7% / 0.2% |
| PQ384B | 96.3% | 95.6% | 84.4% | 89.0% | 58.0% / 70.6% |
| OPQ96B | 84.2% | 83.4% | 0.1% | 0.0% | 0.0% / 0.0% |

All of these settings use a candidate set that contains essentially all exact top-10 neighbors, so exact raw reranking can recover 100% exact R@10.

### Main observation

There is a strong tension:

- aggressive compression makes raw reranking valuable, but the safe bounds are too loose for page pruning;
- very accurate compression makes page pruning and pre-sense highly effective, but compressed-only retrieval is already above 90% exact recall.

PQ384B is the clearest current sweet point if **exact-result preservation is mandatory**.

However, if a 95–96% exact recall compressed-only solution is already acceptable at the application level, the motivation for retaining the raw rerank path becomes weaker.

## 5. Application-quality warning

The exact-ranking gap does not always translate into a similarly large task-quality gap.

For PQ384B, the current quick experiments showed nearly identical nDCG between compressed-only and exact reranking in some settings. Therefore:

- **E-mode** (exact-result preservation) currently has a plausible sweet point;
- **Q-mode** (quality-tolerant retrieval) is less clearly motivated.

This is the most important decision that needs collaborator input.

## 6. RaBitQ status

A quick RaBitQ diagnostic also preserved a large exact-ranking gap while retaining the exact top-k inside the candidate set. However, when evaluated with the same deterministic reconstruction-error bound used for PQ, it produced essentially no safe page-pruning opportunity.

This should **not** be interpreted as evidence that RaBitQ cannot support pruning. The current diagnostic did not yet use RaBitQ-native confidence/distance-bound machinery or a full IVF-RaBitQ v2 implementation.

The next quantization-side question is whether a stronger native bound can move the useful operating point toward lower memory footprints.

## 7. Risk 1 — strong host scheduling may consume pre-sense opportunity

The largest algorithmic risk is that the current pre-sense opportunity is measured before fully optimized host scheduling.

A strong host scheduler can:

- prioritize pages that are most likely to tighten (	heta);
- adapt the submission window (W);
- wait for useful feedback before issuing more pages;
- perform bank-aware ordering;
- prune pages before submission.

If host scheduling moves many pages from

[
t_{	ext{submit}} < t_{	ext{prunable}} < t_{	ext{sense}}
]

to

[
t_{	ext{prunable}} < t_{	ext{submit}},
]

the independent value of device-side pre-sense shrinks.

Therefore the fair comparison must eventually be:

[
	ext{optimized Host-only}
quad	ext{vs}quad
	ext{optimized Host + pre-sense}.
]

## 8. Risk 2 — host cancellation may approximate pre-sense

A stronger baseline is explicit host cancellation.

If a page has already been submitted but has not begun sensing, the host may detect that a tighter (	heta) makes it unnecessary and send a per-page cancel command.

Conceptually:

- **Host cancel:** host identifies the obsolete page and sends `CANCEL(page)`.
- **Pre-sense:** host only publishes the newest (	heta); the device checks necessity locally at the last admission point.

The pre-sense mechanism is only clearly necessary if it has a stable advantage after accounting for a realistic host-cancel path.

The future baseline stack should therefore be:

- **B1:** optimized host scheduling;
- **B1.5:** B1 + explicit host cancel;
- **B2:** B1 + device-local pre-sense recheck.

## 9. Current interpretation

The current experiments support the following conclusions.

- **A1:** raw reranking remains important in lower-accuracy compression regimes.
- **A2:** safe page pruning is highly dependent on bound tightness and shows a sharp threshold effect.
- **A3:** when bounds are tight and many pages are already outstanding, the post-submit / pre-sense opportunity can be large.
- The main unresolved issue is whether that operating point is still practically important once compressed-only recall already exceeds 90%.
- Even if that answer is yes, strong host scheduling and host cancellation remain important threats to the independent contribution of pre-sense.

## 10. Decision questions for the collaboration

Before investing in more complex scheduling or hardware modeling, the collaboration should decide:

1. If compressed-only exact R@10 is already around 90–96%, is exact raw reranking still required in the intended system setting?
2. Is **exact-result preservation** a design requirement, or is small retrieval-quality loss acceptable?
3. If exact preservation is required, is the memory cost of the tighter compressed representation still acceptable?
4. Can RaBitQ-native bounds create a better sweet point at a lower memory footprint?
5. After strong host scheduling and host cancellation, is there still enough residual pre-sense opportunity to justify a device-local mechanism?

The immediate next step should be to resolve Questions 1–2 with the hardware/system collaborator before treating the current PQ384B sweet point as the final target.
