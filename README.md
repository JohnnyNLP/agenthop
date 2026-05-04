# AgentHop

Evaluation harness and construction code for **AgentHop**: a diagnostic benchmark for multi-step scientific question answering. 1,011 four-option MCQs grounded in citation chains over 7,205 arXiv papers from nine major CS venues (2022–2025). Each item is paired with a four-axis decomposition (search recall, conversion rate, tool-use pattern, resource management) that turns a single evaluation run into per-model failure-mode attribution rather than a single accuracy number.

- **Paper:** under review at NeurIPS 2026 Datasets and Benchmarks Track (anonymous)
- **Dataset:** [`agenthop/agenthop` on Hugging Face](https://huggingface.co/datasets/agenthop/agenthop) — CC-BY 4.0
- **Code (this repo):** Apache-2.0

## Repository layout

```
AgentHop/
├── data_loader.py                       # Reference loader for the HF release
├── testbed/                             # ★ Evaluation harness — the main artifact
│   ├── run.py                           # CLI entry point
│   ├── harness.py                       # Async multi-turn tool-use loop
│   ├── tools.py                         # The seven sandboxed tools
│   ├── evaluate.py                      # Four-axis aggregation + summary writer
│   ├── model_configs.py                 # Tool-call parsers + recommended hyperparameters
│   ├── pricing.py                       # Per-token cost table
│   └── MODEL_SUPPORT.md                 # Tool-call support matrix
└── pipeline/                            # All eight construction stages (released for reference)
    ├── stage1_seed_generation.py        # Stage 1: Seed generation
    ├── stage2a_chain_expansion.py       # Stage 2: Chain collection (citation-graph half)
    ├── stage2b_corpus_collection.py     # Stage 2: Chain collection (corpus half)
    ├── stage3a_qa_generation_st.py      # Stage 3: Q/A generation (single-target half)
    ├── stage3b_qa_generation_mt.py      # Stage 3: Q/A generation (multi-target half)
    ├── stage4_distractor_generation.py  # Stage 4: Distractor generation
    ├── stage5_ensemble_filtering.py     # Stage 5: Model-ensemble filtering
    ├── stage6_data_sanity_check.py      # Stage 6: Data sanity check
    ├── stage7_prepare_batch.py          # Stage 7 prep: build LLM-auditor batch JSONL
    ├── stage7_generate_analysis.py      # Stage 7 prep: run the LLM auditor (GPT-5.4 Batch API)
    ├── stage7_prebake_data.py           # Stage 7 prep: slim paper_pool for the Gradio apps
    ├── stage7_quality_app.py            # Stage 7: Gradio interface for the seven-auditor verdict
    ├── stage7_recall_app.py             # Stage 7: Gradio interface for recall-label triage
    ├── stage8_review_app.py             # Stage 8: Gradio review of flagged items (lead-author triage)
    ├── stage8_apply_verdicts.py         # Stage 8: apply audit verdicts back to the release files
    ├── stage8_apply_section_filter.py   # Stage 8: structural section-length post-filter
    ├── build_recall_triage_cases.py     # Helper: build the recall-triage case file
    ├── package_release.py               # Final HuggingFace-format packaging
    └── run_pipeline.py                  # End-to-end runner for the automated stages (2a–6)
    ├── stage7_prepare_batch.py          # Build LLM-auditor batch JSONL
    ├── stage7_generate_analysis.py      # Run the LLM auditor (GPT-5.4 Batch API)
    ├── stage7_prebake_data.py           # Slim paper_pool for the Gradio apps
    ├── stage7_quality_app.py            # Gradio: seven-auditor quality verdict UI
    ├── stage7_recall_app.py             # Gradio: recall-label triage UI
    ├── stage8_review_app.py             # Gradio: lead-author flagged-item triage
    ├── stage8_apply_verdicts.py         # Apply audit verdicts back to release files
    ├── stage8_apply_section_filter.py   # Structural section-length pass
    └── build_recall_triage_cases.py     # Helper: build the recall-triage case file
```

`testbed/` is the main artifact a reviewer needs to reproduce results;
`pipeline/` and `audit/` are released for transparency on how the dataset
was constructed and audited.

## Install

```bash
git clone <this repo>
cd AgentHop
pip install -r requirements.txt
```

API keys are read from environment variables. Set whichever providers you
plan to evaluate:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
export DEEPSEEK_API_KEY=...
export TOGETHER_API_KEY=...   # GLM, Kimi, MiniMax via Together.ai
```

If a `.env` file exists at the repo root and `python-dotenv` is installed,
it is loaded automatically.

## Get the dataset

```bash
pip install huggingface_hub
huggingface-cli download agenthop/agenthop --repo-type dataset \
    --local-dir AgentHop_release
```

This pulls four JSONL files (`qa/full.jsonl`, `graphs/full.jsonl`,
`paper_pool/papers.jsonl`, `audit/recall_labels.jsonl`) plus the loader and
README. From a Python script you can also use the loader directly:

```python
from data_loader import load_agenthop, load_recall_labels

samples = load_agenthop("AgentHop_release")
labels  = load_recall_labels("AgentHop_release")

print(len(samples))                 # 1011
print(samples[0]["question"])
print(samples[0]["paper_pool"])     # papers reachable from this sample's seed
```

## Evaluate a model

```bash
cd testbed

# Frontier API model
python run.py --data ../AgentHop_release --model gpt-4.1 --workers 8

# Anthropic
python run.py --data ../AgentHop_release --model claude-opus-4-6 --workers 4

# Local OpenAI-compatible server (e.g. sglang/vllm)
python run.py --data ../AgentHop_release --model Qwen/Qwen3-32B \
    --base-url http://localhost:8000/v1 --workers 16

# Slice runs
python run.py --data ../AgentHop_release --model gpt-5.4 \
    --question-type multi-target --depth 2

# Closed-book ablation (no retrieval — disables every tool except submit_answer)
python run.py --data ../AgentHop_release --model gpt-5.4 --disable-tools all
```

Each run writes per-sample trajectories to
`experiments/{model}/{sample_id}.json` and a roll-up to
`experiments/{model}/summary.json` containing the four-axis decomposition,
retrieval/conversion stats, tool-call patterns, and cost.

### Resource caps

Per-sample caps (also documented in §3 of the paper):

- **Tool-call budget:** 30 points (navigation tools 1 pt, `read_section` 5 pts, `think`/`submit_answer` 0 pt)
- **Turns:** 20 model-to-tool exchanges
- **Tokens:** 200,000 cumulative prompt + completion

Runs that exhaust any cap before reaching `submit_answer` are recorded as
non-submissions and reported separately.

### The seven tools

| Tool | Cost | What it returns |
|---|---|---|
| `get_paper_info(paper_id)` | 1 | Title, authors, abstract, year |
| `get_references(paper_id)` | 1 | List of cited papers (IDs + titles) |
| `search_papers(query, top_k)` | 1 | Keyword search over titles + abstracts |
| `list_sections(paper_id)` | 1 | Section headers with alphabet aliases |
| `read_section(paper_id, section)` | 5 | Full text of one section |
| `think(reasoning)` | 0 | Private deliberation (no environment effect) |
| `submit_answer(answer, reasoning)` | 0 | Commit to A/B/C/D and end the run |

`read_section` is the only tool that returns paper content. The recall axis
gates on whether the agent called `read_section` on an auditor-labelled
section of every gold paper of the sample.

## The four-axis decomposition

`testbed/evaluate.py` produces a `summary.json` per model that decomposes
accuracy into four metrics:

| Axis | Metric | What it tells you |
|---|---|---|
| **Search** | Section recall — fraction of samples on which the agent read a labelled section of every gold paper | Did it find the evidence? |
| **Synthesis** | Conversion rate — `P(correct \| recall=1)` | Could it use the evidence once it had it? |
| **Tool-use pattern** | Calls per turn (regime indicator, not "higher is better") | Sequential (~1.0) vs parallel-emitting (>1.2) |
| **Resource** | Mean tokens / turns / budget points + non-submission rate | Did it answer at all, and at what cost? |

Aggregate accuracy alone collapses qualitatively different failure modes;
the per-axis attribution surfaces which sub-ability is binding for each
model.

## Reproducing the paper's numbers

Every per-model run writes a `summary.json` to
`experiments/{model}/` with the four-axis breakdown, retrieval and
conversion rates, tool-call patterns, non-submission rate, and per-token
cost. The numbers in Table 2 of the paper come directly from those
`summary.json` files; no extra analysis code is required to reproduce
them.

The closed-book and zero-tool ablations are produced by re-running with
`--disable-tools all`; the off-retrieval condition is produced by passing
the gold paper IDs to the agent (see Appendix C of the paper for the
exact prompt variant).

## Citation

```bibtex
@inproceedings{anonymous2026agenthop,
  title     = {AgentHop: A Diagnostic Benchmark for Multi-Step Scientific Question Answering},
  author    = {Anonymous Authors},
  booktitle = {Submitted to NeurIPS 2026 Datasets and Benchmarks Track},
  year      = {2026}
}
```

Citation will be updated to the published version upon acceptance.

## License

- **Code (this repo):** Apache-2.0 (see `LICENSE`)
- **Dataset (HuggingFace):** CC-BY 4.0
- **Source arXiv content:** as deposited on arXiv (predominantly arXiv-perpetual and CC-BY family)
