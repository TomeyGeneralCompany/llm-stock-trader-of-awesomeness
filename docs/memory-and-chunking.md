# Chunking and tiered memory

The normal event-driven trader and GitHub `trader-sample` probe share one memory implementation. Retrospective reviews and daily feedback also use its bounded context assembly. Inference remains sequential inside the existing host resource lock; there is no embedding server or second model to load.

| Tier | Contents | Lifetime |
| --- | --- | --- |
| Hot working state | Decision date, symbol, current indicators, strategy cash/shares, entry state, events and dominant trend | Replaced by the controller for each decision |
| Warm episodes | Recent accepted decisions from the same run, with original reasoning and conditional expectations | Saved in SQLite; small, explicitly labelled excerpts enter the next prompt |
| Cold evidence | Original strategy file, dated market windows, full technical context, exact prompts and raw model responses | Immutable sources on disk with SHA-256 hashes |

Text is divided into 768-character chunks with 96-character overlap, source identifiers, character offsets, dates and hashes. Market observations are archived in windows of up to 32 rows. The original text remains available; chunks are not model-written summaries. The complete compact strategy policy is mandatory in each trading prompt. Repeated examples and prose in that policy were condensed while retaining its thresholds and entry/exit conditions.

For each decision the controller reserves space for the response and chat framing, installs mandatory policy/current state/question, allocates a small warm-memory allowance, and retrieves dated evidence using keyword overlap and recency. Retrieval streams through SQLite and retains at most 32 candidate chunks in RAM. Future-dated sources and evidence belonging to other runs are excluded from retrieval. This retrieval filter does not replace a separate audit of chronological feature calculation and backtest execution. Exact received market data and gate artifacts remain the authoritative replay inputs.

The context guard uses a conservative UTF-8-byte upper bound rather than an approximate characters-per-token guess. With the default 8192 context and 1500-token trading response allowance, 256 units are reserved for chat framing and a further 900 bytes for the final response task and correction instructions. Oversized mandatory state or questions cause an actionable error before inference. Optional chunks are included only if they fit. This controls model context size; the existing pandas indicator and simulation code still holds its market frame in RAM.

## Decision validation

The populated example JSON that encouraged copied placeholder explanations was removed. The trader requests concrete observations and specific evidence that would change its decision. Both inference and the ANALYZE gate validate the decision fields, finite confidence, and explanations; known placeholder strings are rejected. One correction request is allowed in the same stage. A second invalid answer raises an error; no substitute HOLD is fabricated. This validation does not prove that a model's reasoning is correct.

Each decision log stores a memory receipt: run/date, selected source IDs and offsets, content hashes, original and final prompt hashes/sizes, working-state hash and attempt count. The ANALYZE artifact seals these receipts along with decisions.

## Inspecting a GitHub test

Run **Actions → Ask the Trader → Run workflow**, choosing `trader-sample`. Its summary includes context size, retrieval count and attempt count. Download the **trader-response** artifact for `response.json`, diagnostic logs, and `stage-gates.sqlite3.memory.sqlite3`. The SQLite tables `memory_sources`, `memory_chunks`, `memory_episodes`, and `memory_working` contain the originals and memory records. Each GitHub workflow remains an independent test; it does not import memory from other workflow runs.

Local memory is stored beside `PIPELINE_DB_PATH`, with `.memory.sqlite3` appended. Keep this database together with the gate database when retaining a run. Memory is persistent evidence storage, not automatic execution resume or training of model weights.

| Setting | Default | Purpose |
| --- | --- | --- |
| `MEMORY_CHUNK_CHARS` | `768` | Chunk size, from 128 to 4096 characters |
| `MEMORY_RECENT_EPISODES` | `6` | Recent same-run decisions eligible for warm retrieval |
| `OLLAMA_NUM_CTX` | `8192` | Context window and prompt budget |
| `PIPELINE_DB_PATH` | `data/stage-gates.sqlite3` | Gate database; also determines the memory database location |

No additional Python dependencies are needed. Run `uv run pytest tests -q` for the mocked Ollama integration, stage handoffs, chunk preservation, temporal isolation, integrity, context limits and placeholder regressions. A real Qwen response remains a separate workflow test.

Technical context keeps the full dated history for rolling-indicator warmup while using the short event window for recent price/volume patterns. It preserves prepared adaptive MACD values and interprets `yes`/`no` flags explicitly when classifying market regimes. Named date indexes are handled by the requested observation date.

## Explanations, timeframes and conditional plans

Every new trader decision retains `reasoning` and `expected_outcome` and also requires an `explanation` object:

| Section | Required content |
| --- | --- |
| `observed_state` | Timeframe, actual flat/long position, controller market phase, observed MACD crossover, and specific evidence |
| `interpretation` | Why that evidence supports the present action, including conflicting signals and any distinction between broader and shorter trends |
| `forecast` | A possible future scenario, its timeframe, assumptions, and invalidating evidence |
| `conditional_actions` | Action, trigger, timeframe, required position, prerequisite, and reasoning for each plan |

An uptrend can contain a shorter consolidation when the model explains the timeframes and evidence. A future crossover can be forecast while the observed crossover remains `none`. A future SELL while currently flat is allowed when it requires a long position after entry. The current executable action cannot be SELL with zero shares. These checks preserve conditional forecasting rather than forcing all statements to describe the present.

The controller places observed crossover and compact technical facts in mandatory working context, so optional chunk selection cannot hide them. Inference checks the structured observed facts against that context; the ANALYZE gate independently rechecks them and binds the context to its saved memory hash. One correction attempt includes the actual validation error. Free-text interpretation remains the model's analysis; these structural checks do not prove every sentence correct.

GitHub summaries and existing application decision-log views display the full reasoning sections as readable text. JSON artifacts retain the structured explanation and original model answer. No trading thresholds, stage ordering, or broker integration were changed by this explanation contract.
