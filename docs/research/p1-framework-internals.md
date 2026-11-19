# P1: how DeepEval and Ragas compute and record faithfulness

Issue #13. Pinned versions: `deepeval==4.3.4`, `ragas==0.4.4`, installed in a throw-away venv outside the repo.
All paths are relative to `site-packages/`. **Nothing here was executed** (no judge calls, no stub runs): every
behaviour is a code-reading result. Items marked **UNVERIFIED** rest on third-party code (pydantic, instructor,
langchain) or on a path I did not read.

## 1. Summary: what the final score becomes

Defaults: DeepEval `evaluate()` has `ignore_errors=False`, `skip_on_missing_params=False`
(`deepeval/evaluate/configs.py:46-47`, from env at `deepeval/evaluate/evaluate.py:113-114`).
Ragas `evaluate()` has `raise_exceptions=False` (`ragas/evaluation.py:360`).

| Failure case | DeepEval, native `OllamaModel` | DeepEval, custom `DeepEvalBaseLLM` returning text | Ragas legacy faithfulness via `evaluate()` |
|---|---|---|---|
| Judge returns invalid JSON | pydantic `ValidationError` from inside the model, not retried. Raises out of `evaluate()`; with `IGNORE_DEEPEVAL_ERRORS=1`: `score=None`, `success=False`, `error="ValidationError: ..."` | `ValueError("Evaluation LLM outputted an invalid JSON...")`, same handling | One repair call to the judge, then `OutputParserException`; executor turns it into **NaN**, one `logger.error` line, nothing in the result |
| Empty `retrieval_context` `[]` | Accepted, no guard. Score is whatever the judge says (1.0 if no claims) | same | Accepted, no guard (`""` context). Score is whatever the judge says |
| `retrieval_context=None` | `MissingTestCaseParamsError`: raise, or row skipped (`SKIP_DEEPEVAL_MISSING_PARAMS=1`), or `error` field (`ignore_errors`) | same | Key dropped from row; `KeyError` after the statement call; **NaN** + log |
| Empty answer `""` | `MissingTestCaseParamsError` (same three outcomes) | same | Accepted. If judge returns no statements: **NaN, no log** |
| Whitespace answer `" "` | Accepted. If judge returns no claims: **score 1.0, success True** | same | as `""` |
| Judge response missing a field | Required field missing: `ValidationError` (error row). `reason` missing: fine (Optional) | `verdicts` key missing: **score 1.0**. Item with missing/unknown `verdict`: silently dropped. `claims`/`truths` key missing: `KeyError` | Pydantic error, one repair, then NaN. Empty NLI list: **NaN** + `logger.warning` |
| "Out-of-range score" | Judge never emits a score; verdict must be `yes/no/borderline` (error if not) | Non-members dropped | Judge emits `verdict: int`; **any non-zero int counts as faithful** (2, -1, 7 all = 1) |
| Verdict count != claim count | Not checked | Not checked | Not checked |

Headline: DeepEval can emit **1.0 with `success=True`** when the judge extracts zero claims or returns no usable
verdicts. Ragas emits **NaN** and its default mean drops those rows silently. Neither stores the raw judge text.

## 2. DeepEval faithfulness: prompts and steps (Q1)

Class `deepeval/metrics/faithfulness/faithfulness.py`. Default `threshold=0.5` (`:75`), `async_mode=True` (`:78`),
so `measure()` just runs `a_measure()` (`:125-133`). Steps (sync `:141-148`, async `:182-188`):

1. `prepare_measure` validates the test case (`deepeval/metrics/utils/test_case.py:229-264`).
2. **Truths** from the context: `faithfulness/templates/generate_truths.txt`, passages joined with `"\n\n"`
   (`:329-346`), schema `Truths` (`schema.py:15-16`).
3. **Claims** from the answer: `generate_claims.txt` (`:370-387`), schema `Claims`.
4. **Verdicts**: `generate_verdicts.txt`, one `yes|no|borderline` per claim, `reason` only for no/borderline. The
   evidence is the *truths* list, not the raw context (`:259-299`, `:269`). Skipped (returns `[]`) when no claims (`:262-263`).
5. **Score** = fraction of verdicts in `{yes, borderline}` (`:408-414`; borderline fails only with
   `penalize_ambiguous_claims`), implemented in `deepeval/metrics/utils/qag.py:113-138`. `strict_mode` zeroes a
   score below threshold (`qag.py:138`).
6. **Reason**: fourth judge call, `generate_reason.txt`, unless `include_reason=False` (`:201-257`).

The judge never produces a number; the score is arithmetic over labels. 4 judge calls per row (3 without reason).

## 3. Ragas faithfulness: prompts and steps (Q1)

Two implementations ship in 0.4.4.
- **Legacy** `ragas/metrics/_faithfulness.py`, `ragas.metrics.faithfulness`, accepted by `ragas.evaluate()`.
  Importing it emits a `DeprecationWarning` (`ragas/metrics/__init__.py:224-243`); `evaluate()` is itself deprecated
  (`ragas/evaluation.py:448`).
- **Collections** `ragas/metrics/collections/faithfulness/metric.py`, instructor-based, called as
  `await metric.ascore(user_input, response, retrieved_contexts)` returning `MetricResult`. It rejects legacy LLM
  wrappers (`ragas/metrics/collections/base.py:113-121`).

Legacy steps (`_faithfulness.py:202-214`): (1) `StatementGeneratorPrompt` (`:34-55`, answer split into pronoun-free
statements); **`statements == []` returns NaN** (`:210-211`). (2) `NLIStatementPrompt` (`:73-130`) over
`"\n".join(retrieved_contexts)` (`:157`), one `verdict: int` per statement (`:58-61`). (3) score =
`sum(1 if v.verdict else 0) / len(statements)` (`:184-187`); empty list gives NaN + warning (`:188-192`).
Collections has the same logic (`metric.py:116-163`) but raises `ValueError` for empty `response`, `user_input` or
`retrieved_contexts` (`:103-114`). Full prompt = instruction + JSON signature + few-shot examples + input
(`ragas/prompt/pydantic_prompt.py:120-134`). 2 judge calls per row (plus repairs); no reason is produced.
`max_retries: int = 1` on the metric (`_faithfulness.py:150`) is not passed to `generate` (`:158-162`, `:174-178`), so
the effective parse-retry budget is the default `retries_left=3` (`pydantic_prompt.py:143`).

## 4. Bad judge output in detail (Q2)

### DeepEval
- Dispatch: `generate_with_schema_and_extract` (`deepeval/metrics/utils/generation.py:102-133`): a schema instance is
  used as is, otherwise text goes through `trimAndLoadJson` (`:27-60`).
- **Native `OllamaModel`** validates inside `generate`: `schema.model_validate_json(response.message.content)`
  (`deepeval/models/llms/ollama_model.py:98-103`, async `:127-132`), after sending `format=<JSON schema>` (`:92`).
  Bad JSON or a missing required field raises pydantic `ValidationError` before `trimAndLoadJson`; the raw text is
  discarded. The `@retry_ollama` wrapper (`:78`) retries only timeouts, network errors and 5xx
  (`deepeval/models/retry_policy.py:306-330`), default 2 attempts (`:20`). Validation errors are not retried.
- **Custom model** returning a string: `trimAndLoadJson` takes first `{` to last `}`, strips trailing commas, else
  raises `ValueError` and sets `metric.error` (`generation.py:37-58`). `None` raises too (`:31-35`).
- The loose path is lenient. `_verdicts_from_json` uses `data.get("verdicts", []) or []` (`qag.py:106`) and drops
  items whose verdict is outside the vocabulary (`qag.py:107-109`). It keeps only the leading word, so "No, it is not"
  counts as `no`, and `idk` maps to borderline (`qag.py:58-81`, `base_metric.py:69`). **Key missing, or every item
  dropped, gives an empty list and `score_qag_verdicts` returns 1** (`qag.py:118`, `:129-130`). Claims/truths use
  `data["claims"]` / `data["truths"]` (`faithfulness.py:345,367,386,405`): missing key is a bare `KeyError`.
- `evaluate()` error handling: `_execute_metric` (`deepeval/evaluate/execute/_common.py:236-292`). With
  `ignore_errors` it sets `metric.error = "<Type>: <msg>"`, `success=False`, continues; otherwise re-raises. The error
  goes to `MetricData.error` and `score` is forced to `None` (`deepeval/evaluate/utils.py:99-115`). This code path
  logs nothing itself.
- **Skipped rows**: only `MissingTestCaseParamsError` with `skip_on_missing_params`. The metric is flagged
  `skipped` and `continue`d, so it is **absent from `metrics_data`** with no error recorded (`_common.py:256-260`,
  `deepeval/evaluate/execute/e2e.py:294-306`).
- Input checks: `check_llm_test_case_params` (`test_case.py:148-226`): `actual_output == ""` raises (`:202-207`);
  any required param `None` raises (`:209-226`). `[]` context is not `None` and the `LLMTestCase` validator accepts any
  list (`deepeval/test_case/llm_test_case.py:539-548`), so it passes.
- **Stale score hazard**: neither `prepare_measure` nor `measure` resets `self.score` (`test_case.py:229-264`,
  `faithfulness.py:113-159`). A hand-rolled loop that reuses one metric and catches exceptions reads the previous
  row's score for a failed row. `evaluate()` is safe (it nulls the score on error).
- Error rows have `success=False` (`deepeval/metrics/base_metric.py:172-182`): an exporter that reads only `success`
  counts an error as a judge "fail".

### Ragas (legacy)
- Parse path depends on the LLM type (`ragas/prompt/pydantic_prompt.py:289-322`):
  - `LangchainLLMWrapper` or other `BaseRagasLLM`: `RagasOutputParser.parse_output_string` (`:526-558`) extracts JSON
    (`ragas/prompt/utils.py:69-106`); on failure it asks the judge to reformat (one repair, `:538-555`) and parses
    again with no handler (`:555`), so a second failure raises langchain `OutputParserException`.
    `RagasOutputParserException` is raised only when `retries_left == 0` (`:556-557`).
  - Instructor LLM or langchain-native LLM: `model_validate_json` directly, **no retry** (`:317-319`), pydantic
    `ValidationError`.
- The executor catches every exception per job. With `raise_exceptions=False` it logs
  `"Exception raised in Job[%s]: ..."` at ERROR (no traceback) and returns `np.nan` (`ragas/executor.py:64-86`).
  With `True` it re-raises (`:72-73`).
- Transport retries: up to `RunConfig.max_retries=10`, timeout 180 s (`ragas/run_config.py:51-59,97-115`). Judge
  temperature defaults to 0.01 (`ragas/llms/base.py:83-85,102`).
- If **every** job returns nothing: `ExceptionInRunner` (`ragas/evaluation.py:288-289`). Partial NaNs never raise.
- Verdict range: `verdict: int` (`_faithfulness.py:61`) scored by truthiness (`:184-186`), so 2 or -1 count as
  faithful. A string like `"yes"` fails int validation; edge coercions (`"1"`, `1.0`, `0.5`) are **UNVERIFIED** (pydantic).
- NaN is invisible downstream: the result's mean uses `safe_nanmean` (`ragas/dataset_schema.py:445`,
  `ragas/utils.py:46-55`) and prints no dropped-row count.
- Collections: failures surface as plain exceptions to the caller (`metric.py:103-114`). Instructor retry behaviour
  is inside the third-party library (**UNVERIFIED**). Default `max_tokens=1024` (`ragas/llms/base.py:781-783`) might
  truncate long NLI output (**UNVERIFIED inference**).

## 5. What the exports contain (Q3)

### DeepEval
- `evaluate()` returns `EvaluationResult.test_results: List[TestResult]` (`deepeval/evaluate/types.py:11-35`). Per test
  case: `name, success, input, actual_output, expected_output, context, retrieval_context, index, metadata,
  metrics_data` (`deepeval/evaluate/utils.py:219-272`).
- Per metric `MetricData` (`deepeval/tracing/api.py:64-82`): `name, threshold, success, score, reason, strictMode,
  evaluationModel, error, evaluationCost, inputTokenCount, outputTokenCount, verboseLogs`.
- `verboseLogs` is **always populated** on success, even with `verbose_mode=False` (`faithfulness.py:149-157`,
  `deepeval/metrics/utils/verbose.py:38-50`): parsed truths, claims, verdicts (with reasons). Strongest evidence in
  the export, but parsed objects, not raw judge text. `None` on error rows.
- Persistence: `.deepeval/.latest_run_full.json` plus a timestamped file in `DEEPEVAL_RESULTS_FOLDER`
  (`deepeval/test_run/test_run.py:1065-1090`; `deepeval/config/settings.py:365-377`).
- **Raw judge output is stored nowhere.** Native Ollama discards it after validation.
- **Capture**: pass a custom `DeepEvalBaseLLM` whose `generate`/`a_generate` call Ollama directly, append
  `(prompt, raw_text)` to a per-row log, and return the **raw string**; the string path is supported
  (`generation.py:130-133`) and gives the `ValueError`/`error` route on bad JSON. `generate_with_schema` forwards
  `schema=` and retries without it on `TypeError` (`deepeval/models/base_model.py:163-169`). Non-native models skip
  cost tracking (`generation.py:126-129`), irrelevant for Ollama (cost 0, `ollama_model.py:104`).

### Ragas
- `evaluate()` returns `EvaluationResult` (`ragas/dataset_schema.py:412-460`): `scores` (list of
  `{metric: float|NaN}`), `dataset`, `traces`, `cost_cb`, `run_id`. `to_pandas()` = dataset columns + score columns
  (`:462-496`). No error column, no reason, no verdict detail.
- `result.traces` (`ragas/callbacks.py:134-173`): per row, metric, prompt: `{input, output}` with the **parsed pydantic
  output** (statements; per-statement verdict + reason). `RagasTracer` is seen overriding `on_chain_start/end`
  only (`callbacks.py:83-132`; **UNVERIFIED** that no later error handler exists), so a failed prompt likely leaves
  empty outputs and no error text.
- Raw judge text is not stored. Capture: subclass `LangchainLLMWrapper` (or `BaseRagasLLM`) and record in
  `agenerate_text` (`ragas/llms/base.py:291-`) before the parser runs. Langchain `callbacks=` to `evaluate` for
  `on_llm_end` is an alternative (**UNVERIFIED**).
- Failure signal = NaN in `scores` + a Python log line. If logs are not captured, the cause is lost.

## 6. Local Ollama and telemetry (Q4)

**DeepEval + Ollama**
- Pass `model=OllamaModel(model="<tag>", base_url="http://localhost:11434", temperature=0)` to the metric; it counts
  as native (`deepeval/metrics/utils/models.py:239-256`). Defaults: that base URL and temperature 0.0
  (`ollama_model.py:40-52`).
- Env route: `LOCAL_MODEL_API_KEY=ollama` (`deepeval/metrics/utils/models.py:90-100`) with `OLLAMA_MODEL_NAME` /
  `LOCAL_MODEL_BASE_URL` (`deepeval/config/settings.py:679-721`).
- Needs the `ollama` package at call time (`ollama_model.py:216-220`): **new dependency, needs approval** (charter).
- **Cost trap**: with no `model`, `_build_model` falls back to `OpenAIModel` (`deepeval/metrics/utils/models.py:231`).

**Ragas + Ollama**
- Collections: `llm_factory("<tag>", provider="openai", client=AsyncOpenAI(base_url="http://localhost:11434/v1",
  api_key="ollama"))` (`ragas/llms/base.py:624-760`); `openai` and `instructor` are ragas dependencies.
- Legacy: `LangchainLLMWrapper(ChatOllama(...))` needs `langchain-ollama`, not a ragas dependency (**UNVERIFIED**).
- **Cost trap**: `evaluate()` with `llm=None` builds `OpenAI()` + `gpt-4o-mini` (`ragas/evaluation.py:176-179`);
  embeddings default is also inferred from it (`:183-195`). Always pass `llm=` and `embeddings=`.

**Phone-home inventory**

| Framework | Network call | Trigger | Disabled by the env var? |
|---|---|---|---|
| DeepEval | PostHog `https://us.i.posthog.com` (`deepeval/telemetry/client.py:25-26,94-97`) | on by default | Yes. `DEEPEVAL_TELEMETRY_OPT_OUT=YES` parses truthy (`deepeval/config/utils.py:17-18`, validator `settings.py:1451-1500`); `capture` returns (`client.py:179-180`); backend is `NoopBackend` (`:122-123`) |
| DeepEval | Anonymous id file in `~/.deepeval` | first event | Yes, writes skipped (`telemetry/identity.py:110-112,150-152`) |
| DeepEval | PyPI check `pypi.org/pypi/deepeval/json` (`__init__.py:111-144`) | only if `DEEPEVAL_UPDATE_WARNING_OPT_IN=1` | Off by default |
| DeepEval | Confident AI upload (`test_run/test_run.py:1245-1246`) | only if a Confident API key is set (`confident/api.py:146-147`) | Not triggered without a key |
| Ragas | `POST https://t.explodinggradients.com`, 1 s timeout (`ragas/_analytics.py:37-38,233`) | on by default. Events: evaluation (metric names, row count), `llm_usage` (**provider and model name**, `ragas/llms/base.py:758-`), prompt_usage | Yes, but only the exact value `true` (case-insensitive) works (`:49`); `1` or `yes` do **not**. Value is cached at first call (`:45`): set it before importing ragas |
| Ragas | Local `uuid.json` in the `ragas` user data dir | `get_userid` runs on event construction (`:81-92,111`) before `track` checks the flag (`:224`) | **No** (local file only, no network). **UNVERIFIED** by execution |
| Ragas | Background flush thread + atexit hook | import time (`:287-289`) | Thread runs, `track` returns early |

`DEEPEVAL_TELEMETRY_OPT_OUT=YES` plus `RAGAS_DO_NOT_TRACK=true` stop every analytics network call I found. Not
audited: DeepEval tracing/`@observe` integrations, CLI `login`, benchmark dataset downloads, Ragas `helicone_config`
and testset generation; JudgeCheck should not call them. Set both vars before importing either library, and in CI
restrict egress to `localhost:11434` as defence in depth.

## 7. Implications for JudgeCheck

**Detectable from exports alone**
- DeepEval: rows with `error != null` or `score == null` (the type-name prefix of `error` separates
  `ValidationError`, `ValueError`, `MissingTestCaseParamsError`). Never rely on `success` alone: error rows carry
  `success=False`.
- DeepEval: **rows missing from `metrics_data`** (skip mode), found by joining on example ID; the export does not mark them.
- DeepEval: **suspicious 1.0**: `score == 1.0` with a blank `actual_output`, or `verboseLogs` showing empty
  `Claims:` or `Verdicts:`. The silent pass is visible only through `verboseLogs`.
- Ragas: NaN in `scores`, and NaN count vs row count (the headline mean hides it).
- Both: `[]`/None context and empty/whitespace answers are visible in the input columns and are not flagged by
  either framework, so JudgeCheck must flag them (and never average them in).
- Ragas: per-statement integer verdicts outside `{0,1}` appear in `traces`. Claim/statement count vs verdict count
  mismatch is visible in DeepEval `verboseLogs` and Ragas `traces`.

**Needs raw-output capture** (a judge wrapper that logs prompt + raw response per call)
- The cause of a Ragas NaN (log line only) and the full DeepEval error text.
- Whether bad JSON was repaired (Ragas repair call is invisible) or items were dropped (DeepEval loose path, no trace).
- A missing `verdicts` key yielding 1.0, and borderline verdicts counted as passes.
- Truncation vs genuinely malformed JSON; the exact prompt sent (prompt versioning for RunMeta).
- Probes (D5) call the metrics directly, so the wrapper sees every call there.

**Recommendations**
1. Use the recording `DeepEvalBaseLLM` wrapper (not native `OllamaModel`) for probes so raw text lands in findings.
2. Treat `1.0` and NaN as untrusted until the claim/statement count is known to be non-zero.
3. Record `IGNORE_DEEPEVAL_ERRORS`, `SKIP_DEEPEVAL_MISSING_PARAMS` and `raise_exceptions` in `RunMeta`: they decide
   which failures exist in an export.
4. Never claim "no parse errors" from a DeepEval export made without `ignore_errors` (it would have raised) or with
   `skip_on_missing_params` (skipped rows vanish).
