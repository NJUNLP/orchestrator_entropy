# IWG — Inverse Workflow Generation

[Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems](https://arxiv.org/abs/xxxx). Junze Zhu, Weihao Chen, Xuanwang Zhang, Zhen Wu, Xinyu Dai. In Proceedings of ICML, 2026.

IWG is a multi-agent pipeline that synthesizes process-verifiable benchmarks by reconstructing execution environments backward from target solutions. It enables dense, step-level measurement of orchestrator scheduling entropy for mean-field dynamics analysis.

## Setup

```bash
pip install openai
```

Configure your API key in `iwg/config.json`:

```json
{
  "model": {
    "api_key": "sk-...",
    "model": "gpt-4.1"
  }
}
```

## Data

Each Seed Data declares the query and groundtruth.

```json
{
  "id": "seed_001",
  "query": "What year was the director of The White Ribbon born?",
  "answer": "March 23, 1942",
}
```

## Usage

The pipeline is fully decoupled: generate benchmarks first, then run any orchestrator model independently.

### Phase 1 — Generate benchmarks

```bash
python3 -m iwg.generate_benchmarks                    # all seeds
python3 -m iwg.generate_benchmarks --seed-id seed_001  # single seed
```

Benchmarks are saved to `bench/` as static JSON files (environments, checkpoints, gold agent sequences).

### Phase 2 — Run orchestrator

```bash
python3 -m iwg.run_orchestrator --list                          # list benchmarks
python3 -m iwg.run_orchestrator --bench 001 --model gpt-4.1  # single run
python3 -m iwg.run_orchestrator --all --model gpt-4.1        # all benchmarks
```

Trajectories and metrics are saved to `trajectories/` independently. Run the same benchmark with different models for direct comparison:

```bash
python3 -m iwg.run_orchestrator --bench 001 --model gemini-2.5-pro,gpt-4o,claude-sonnet-4-6
```

### Metrics

Six trajectory-aware metrics (LCS-F1, Task Success, Step Success Rate, Exception Handling F1, Faithfulness, Consistency) are computed automatically. See `iwg/metrics.py`.



## Citation

```bibtex
@inproceedings{zhu2026recognize,
  title     = {Recognize Your Orchestrator: An Entropy Dynamics Perspective
               for LLM Multi-Agent Systems},
  author    = {Zhu, Junze and Chen, Weihao and Zhang, Xuanwang and
               Wu, Zhen and Dai, Xinyu},
  booktitle = {Proceedings of the 43rd International Conference on
               Machine Learning},
  year      = {2026},
}
```
