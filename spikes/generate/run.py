"""Spike: run DeepEval / Ragas faithfulness on a local Ollama judge and capture everything.

Capture method: DeepEval via a thin subclass of its OllamaModel (overrides generate and
a_generate); Ragas via httpx event hooks on the OpenAI-compatible client it is handed.
No monkeypatching of framework internals.
"""

import os

# Telemetry off BEFORE the frameworks are imported.
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import argparse  # noqa: E402
import asyncio  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from datetime import UTC, datetime  # noqa: E402
from importlib.metadata import version  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

import httpx  # noqa: E402
import pandas  # noqa: E402

OLLAMA = "http://localhost:11434"
TEMPERATURE = 0.0
SEED = 0
HERE = Path(__file__).parent


class Recorder:
    """Appends raw judge request/response records to raw_calls.jsonl."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.write_text("", encoding="utf-8")
        self.n = 0
        self.example_id: Any = None  # item being processed; items run sequentially

    def add(self, record: dict[str, Any]) -> None:
        self.n += 1
        record = {
            "call": self.n,
            "example_id": self.example_id,
            "ts": datetime.now(UTC).isoformat(),
            **record,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _record(rec: Recorder, resp: httpx.Response) -> None:
    rec.add(
        {
            "request": resp.request.extensions.get("rec"),
            "url": str(resp.request.url),
            "status": resp.status_code,
            "response": resp.json() if resp.status_code == 200 else resp.text,
        }
    )


def _hooks(rec: Recorder) -> dict[str, list[Any]]:
    """httpx event hooks (async) that record each raw request/response pair."""

    async def on_request(req: httpx.Request) -> None:
        req.extensions["rec"] = json.loads(req.content or b"null")

    async def on_response(resp: httpx.Response) -> None:
        await resp.aread()
        _record(rec, resp)

    return {"request": [on_request], "response": [on_response]}


def _sync_hooks(rec: Recorder) -> dict[str, list[Any]]:
    """Same as _hooks, for synchronous httpx clients."""

    def on_request(req: httpx.Request) -> None:
        req.extensions["rec"] = json.loads(req.content or b"null")

    def on_response(resp: httpx.Response) -> None:
        resp.read()
        _record(rec, resp)

    return {"request": [on_request], "response": [on_response]}


def _deepeval_judge(client: str, model: str, rec: Recorder) -> Any:
    from deepeval.models import LocalModel, OllamaModel
    from openai import AsyncOpenAI

    if client == "native":

        class RecordingOllama(OllamaModel):  # type: ignore[misc]
            """Native OllamaModel; its client kwargs are forwarded to httpx, so we add hooks."""

            def _build_client(self, cls: Any) -> Any:
                hooks = _hooks(rec) if cls.__name__ == "AsyncClient" else _sync_hooks(rec)
                return cls(host=self.base_url, event_hooks=hooks)

        return RecordingOllama(
            model=model,
            base_url=OLLAMA,
            temperature=TEMPERATURE,
            generation_kwargs={"seed": SEED},
        )

    class RecordingLocal(LocalModel):  # type: ignore[misc]
        """LocalModel (OpenAI-compatible Ollama endpoint) with a recording http client."""

        def load_model(self, async_mode: bool = False) -> Any:
            http = httpx.AsyncClient(event_hooks=_hooks(rec))
            return AsyncOpenAI(
                api_key="ollama", base_url=self.base_url, http_client=http, max_retries=0
            )

        def generate(self, prompt: str, schema: Any = None, audio: Any = None) -> Any:
            return asyncio.run(self.a_generate(prompt, schema, audio))

    return RecordingLocal(
        model=model,
        api_key="ollama",  # placeholder; Ollama ignores it
        base_url=f"{OLLAMA}/v1",
        temperature=TEMPERATURE,
        generation_kwargs={"seed": SEED},
    )


def run_deepeval(
    items: list[dict[str, Any]], model: str, out: Path, rec: Recorder, client: str = "native"
) -> None:
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig, DisplayConfig
    from deepeval.metrics import FaithfulnessMetric
    from deepeval.test_case import LLMTestCase

    judge = _deepeval_judge(client, model, rec)
    by_id = {str(i["id"]): i["id"] for i in items}

    class TaggedFaithfulness(FaithfulnessMetric):  # type: ignore[misc]
        """Sets the current item id before each (sequential) measure() so raw calls are tagged."""

        def measure(self, test_case: Any, *args: Any, **kwargs: Any) -> Any:
            rec.example_id = by_id[test_case.name]
            return super().measure(test_case, *args, **kwargs)

    # DeepEval resolves prompt templates by class name
    TaggedFaithfulness.__name__ = "FaithfulnessMetric"
    metric = TaggedFaithfulness(model=judge, async_mode=False, include_reason=True)
    cases = [
        LLMTestCase(
            name=str(i["id"]),
            input=i["question"],
            actual_output=i["answer"],
            retrieval_context=[i["context"]],
        )
        for i in items
    ]
    result = evaluate(
        cases,
        [metric],
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(show_indicator=False, print_results=False),
    )
    (out / "native_deepeval.json").write_text(
        result.model_dump_json(indent=2, by_alias=True), encoding="utf-8"
    )


def run_ragas(items: list[dict[str, Any]], model: str, out: Path, rec: Recorder) -> None:
    from openai import AsyncOpenAI
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import llm_factory
    from ragas.metrics import Faithfulness

    http = httpx.AsyncClient(event_hooks=_hooks(rec))
    # Ollama's OpenAI-compatible endpoint; the key is a required-but-ignored placeholder.
    client = AsyncOpenAI(base_url=f"{OLLAMA}/v1", api_key="ollama", http_client=http)
    llm = llm_factory(model, provider="openai", client=client, temperature=TEMPERATURE, seed=SEED)
    frames = []
    for i in items:  # one evaluate() per item so every raw call is tagged with its id
        rec.example_id = i["id"]
        dataset = EvaluationDataset.from_list(
            [
                {
                    "user_input": i["question"],
                    "response": i["answer"],
                    "retrieved_contexts": [i["context"]],
                }
            ]
        )
        df = evaluate(dataset, [Faithfulness()], llm=llm, show_progress=False).to_pandas()
        df.insert(0, "example_id", i["id"])
        frames.append(df)
    df = pandas.concat(frames, ignore_index=True)
    df.to_csv(out / "native_ragas.csv", index=False)
    df.to_json(out / "native_ragas.json", orient="records", indent=2)


def ollama_digest(model: str) -> str | None:
    r = httpx.post(f"{OLLAMA}/api/show", json={"model": model}, timeout=30)
    r.raise_for_status()
    tags = httpx.get(f"{OLLAMA}/api/tags", timeout=30).json()["models"]
    return next((m["digest"] for m in tags if m["name"] == model), None)


def _with_framework(argv: list[str], fw: str) -> list[str]:
    out = list(argv)
    out[out.index("--framework") + 1] = fw
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--items", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--framework", choices=["deepeval", "ragas", "both"], required=True)
    p.add_argument("--deepeval-client", choices=["native", "openai-compat"], default="native")
    p.add_argument("--limit", type=int, default=None, help="use only the first N items")
    args = p.parse_args()
    if args.framework == "both":  # one process per framework: DeepEval leaves Ragas hanging
        for fw in ("deepeval", "ragas"):
            subprocess.run(  # noqa: S603
                [sys.executable, __file__, *_with_framework(sys.argv[1:], fw)], check=True
            )
        return
    items = json.loads(args.items.read_text(encoding="utf-8"))[: args.limit]
    slug = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    runners = {
        "deepeval": lambda *a: run_deepeval(*a, client=args.deepeval_client),
        "ragas": run_ragas,
    }
    for fw in ("deepeval", "ragas") if args.framework == "both" else (args.framework,):
        out = HERE / "out" / f"{fw}-{slug}"
        out.mkdir(parents=True, exist_ok=True)
        rec = Recorder(out / "raw_calls.jsonl")
        runners[fw](items, args.model, out, rec)
        meta = {
            "framework": fw,
            "framework_version": version(fw),
            "model": args.model,
            "ollama_digest": ollama_digest(args.model),
            "temperature": TEMPERATURE,
            "seed": SEED,
            "n_items": len(items),
            "deepeval_client": args.deepeval_client if fw == "deepeval" else None,
            "n_raw_calls": rec.n,
            "timestamp": datetime.now(UTC).isoformat(),
            "versions": {k: version(k) for k in ("deepeval", "ragas")},
        }
        (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"{fw}: wrote {out} ({rec.n} raw calls)")


if __name__ == "__main__":
    main()
