# Reproducibility re-run (#38)

Ragas faithfulness, `qwen2.5-coder:3b`, `data/golden/v0/items.json` (30 items), temperature 0, seed 0,
Ollama 0.40.16, ragas 0.4.4. Command:

    uv run python run.py --items ../../data/golden/v0/items.json --model qwen2.5-coder:3b \
      --framework ragas --out-root out/repro

Output: `out/repro/ragas-qwen2.5-coder_3b/` (meta.json now records `items_sha256` and `ollama_version`).

Result vs committed `out/ragas-qwen2.5-coder_3b/`: all 30 scores identical (0 items differ, max difference 0).
No verdict changes at threshold 1.0, so D7 (93% agreement, 1 false pass) is unchanged.
