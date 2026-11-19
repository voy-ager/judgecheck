# Spike: DeepEval and Ragas faithfulness on a local judge

Isolated uv project (own `pyproject.toml` and `uv.lock`); it is NOT part of the core lockfile and never ships.
Needs Ollama running on `localhost:11434` with the judge model pulled. No API keys, no other network.

```
cd spikes/generate
uv sync
uv run python run.py --items ../../scripts/judge_smoke_items.json \
  --model qwen2.5-coder:3b --framework both --limit 3
```

`--framework` is `deepeval`, `ragas` or `both`; `--limit N` uses the first N items (omit for all).
Output goes to `out/<framework>-<model>/` (`:` becomes `_`): the framework's native export
(`native_deepeval.json`, or `native_ragas.csv` and `.json`), `raw_calls.jsonl` (every raw judge
request and response, recorded by an httpx event hook on the judge's HTTP client) and `meta.json`
(model, Ollama digest, framework versions, temperature 0, seed 0, timestamp).
`--deepeval-client native|openai-compat` (default native) picks DeepEval's own Ollama client or its OpenAI-compatible LocalModel.
Telemetry is switched off in `run.py` before the frameworks are imported.
