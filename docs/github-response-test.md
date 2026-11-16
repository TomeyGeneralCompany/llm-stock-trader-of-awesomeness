# Test Responses from GitHub

Use **Actions → Ask the Trader → Run workflow** in this repository. Select the `main` branch, choose a mode, enter your question, and start the run. Open the run's **Summary** when it finishes to read the response.

| Mode | What it exercises |
| --- | --- |
| `trader-sample` | One decision using the current strategy's prompt builder, technical analysis, decision principles, local Qwen client, and decision-output validation. It uses a clearly labeled synthetic `SAMPLE` instrument with 60 daily observations. |
| `model-question` | A free-form question through the same local Qwen client. No market data is supplied. |

The runner installs Ollama and loads `qwen3:4b`. Responses come from model inference; the workflow does not use the mocked responses used by the automated tests. All computation occurs on GitHub's runner, so your work computer only needs the GitHub website. Initial setup and model download can take several minutes.

Each run is a separate question. This is a response probe: it does not submit orders or run the full historical backtest. The sample decision uses synthetic prices, and runner timing is not a measurement of your laptop.

The run summary identifies the mode, model tag, commit, question, data source, and answer. The `trader-response` artifact contains the Markdown and JSON transcripts, execution log, and available Ollama diagnostics. Artifacts are retained for seven days.

This repository is public, so questions, answers, and run logs are visible in its public Actions history.

Example sample-mode questions:

- Explain your decision and what evidence would change it.
- Which technical indicators support your decision, and which conflict?
- How do the existing strategy principles affect this sample decision?

To run the same probe on a local installation:

```bash
PYTHONPATH=backend TRADER_TEST_MODE=model-question TRADER_TEST_PROMPT="Explain moving averages." uv run python -m app.ask_trader
```

The sample also exercises [chunked evidence and tiered memory](memory-and-chunking.md). Its artifact includes original sources and exact prompts in SQLite. Placeholder explanations receive one correction attempt and cannot pass as completed decisions.
