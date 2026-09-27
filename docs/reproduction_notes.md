# Reproduction notes

The scripts under `experiments/` are the validated research snapshots used for the current feasibility study. They intentionally preserve the experiment logic and parameter choices that produced the reported results.

## Important path assumptions

The current snapshots were originally executed from a research workspace and therefore contain absolute path assumptions such as `/workspace/HBF_RAG_Sim` and `/opt/conda/bin/python`.

For reproduction on another machine:

1. replace the workspace root with a local directory;
2. use any Python environment satisfying `requirements.txt`;
3. download BEIR SciFact and FiQA into the expected `data/<dataset>/` layout;
4. let Hugging Face cache the requested sentence-transformer model locally;
5. do not interpret the discrete-event timing numbers as real HBF hardware measurements.

## Current experiment layers

- `hbf_rag_quick_validate_v1.sh`: synthetic mechanism sanity check;
- `g1_quick_real_v1.sh`: real-data compressed-only vs exact-rerank necessity and timing decomposition;
- `bound_sensitivity_v1.sh`: code length / candidate-budget sensitivity;
- `sweetspot_768_v1.sh`: 768-d sweet-spot validation;
- `rabitq_768_v1.sh`: RaBitQ diagnostic using the currently available deterministic reconstruction-bound path.

The RaBitQ diagnostic is not yet a full IVF-RaBitQ v2 evaluation and does not use RaBitQ-native confidence bounds.
