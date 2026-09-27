# HBF_RAG

Research prototype and diagnostic materials for studying feedback-aware page admission and pre-sense pruning for HBF-backed vector reranking in RAG.

## Current research question

The current work asks whether an HBF page that was necessary when submitted can become safely unnecessary before `sense_start`, after exact-distance feedback tightens the query's top-k threshold.

The present evidence shows a conditional sweet spot:

- aggressive compression makes raw-vector reranking important, but the deterministic safe distance bounds are too loose to prune pages;
- high-accuracy compression makes the bounds tight enough for substantial page pruning and pre-sense savings, but compressed-only retrieval is already above 90% exact recall;
- therefore the key open question is whether preserving exact reranking is still worth optimizing once compressed-only recall is already above 90%.

A second open question is whether strong host-side scheduling or host cancellation consumes most of the post-submit/pre-sense opportunity before a device-local mechanism is needed.

See [docs/current_findings_20260927.md](docs/current_findings_20260927.md).

## Repository layout

- `experiments/hbf_rag_quick_validate_v1.sh`: first synthetic feasibility prototype.
- `experiments/g1_quick_real_v1.sh`: real SciFact/FiQA G1 quick validation and timing decomposition.
- `experiments/bound_sensitivity_v1.sh`: code-length and candidate-budget sensitivity.
- `experiments/sweetspot_768_v1.sh`: 768-d PQ/OPQ sweet-spot experiment.
- `experiments/rabitq_768_v1.sh`: 768-d RaBitQ quick diagnostic.
- `results/summary_20260927.json`: compact snapshot of the current key results.
- `docs/current_findings_20260927.md`: current analysis, conclusions, risks, and next-step decision criteria.

## Interpretation boundary

These experiments are feasibility and mechanism diagnostics. The timing model is a parameterized discrete-event model, not a claim of measured HBF hardware performance. The strongest current pre-sense numbers are intentionally measured before optimized host scheduling and host cancellation, so they are an upper-bound opportunity signal rather than the final SHARE-RAG gain.
