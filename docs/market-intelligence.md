# Continuous market intelligence

This layer extends the existing local paper trader. It has no brokerage credentials, real-order API, portfolio-write permission or independent execution function. The existing regular-session window, $1-to-$100 challenge objective, Qwen/Ollama configuration, paper validation and accumulated portfolio databases remain in place. The separately authorized historical-engine fixes are described in execution-and-accounting.md.

## Data flow and persistence

Public sources and market observations enter an independent SQLite archive, then full-text/entity indexing, evidence-backed causal investigations, measurable forecasts and opportunity ranking. The paper controller receives bounded catalyst/relationship/lesson context while retaining its own technical ranking, holdings and pending-order priority. Only that controller can authorize simulated orders.

The intelligence database adds sources, acquisition_jobs, securities, market_observations, documents, document_versions, document_chunks, entities, entity_relationships, events, event_evidence, research_tasks, research_findings, causal_hypotheses, market_forecasts, forecast_outcomes, learned_lessons and coverage. Additional audit/scheduler tables retain exact research prompts/responses and control state. Migrations are additive; no portfolio database is reset or repurposed.

Original text and payloads, normalized URL, publisher/title/language, content hash, publication/acquisition/revision/event times and available time are retained. A document's available time is no earlier than either publication or acquisition. Changed content creates a separate version; overlapping chunks retain original-text offsets and hashes. Headlines can cluster events, but similar or syndicated reports never erase distinct source evidence or contradictions. Publishing domains are provenance, not a guarantee that publishers are independent.

Retrieval uses SQLite full text, timestamps and entity relationships. Originals remain on disk; only bounded chunks and relevant prior findings enter active context. The existing shared stage/inference lock permits one heavy workload at a time. No embedding model or GPU is introduced. External text is untrusted evidence; model outputs cannot request a terminal or arbitrary commands.

## Acquisition

Default polling: SEC EDGAR, Federal Reserve, BLS, BEA and RSS every five minutes; Treasury/GDELT/corporate IR feeds every fifteen minutes when configured; EIA every thirty minutes; Nasdaq listings daily; Yahoo market jobs recur continuously. FRED/ALFRED adapters are enabled only with an optional FRED_API_KEY. INTELLIGENCE_FEEDS accepts a JSON array of id/url/interval objects for public RSS or corporate investor-relations feeds. Corporate sites without configured public feeds remain missing coverage, not fabricated reports.

SEC requests require SEC_USER_AGENT identifying the operator/contact. A missing identity is a recorded, retryable source configuration failure. Network reads have finite connect/read and response-size bounds, public-destination checks, controlled redirects and retries. Source outages remain visible and do not prevent unrelated jobs. Leases, fair scheduling, cursors and pending jobs survive restart. Queue processing per poll is bounded; the total investigation has no arbitrary wall-clock cutoff.

Nasdaq metadata seeds the security catalog and pending coverage. Yahoo enriches issuer metadata and archives daily and five-minute observations; extended-session observations are labelled separately and never become authorized paper fills. Configured sector, commodity, yield, currency and market benchmarks are observations rather than assumed tradable instruments. Provider limitations and missing history are shown in coverage. The archive grows from actual successful acquisitions; it does not claim instantaneous full-universe coverage.

## Research and learning

Investigations retain questions, completed steps, dependencies and errors. Qwen must supply supported findings, supporting/contradictory status, cited entity relationships, causal hypotheses and follow-ups. Every cited exact quotation is checked against an available original-text range. Supported relationship predicates include supplies, purchases, competes, owns, operates_in, regulated_by, depends_on, exposed_to and financed_by. Affected securities must be registered and grounded in cited identity evidence.

Hypotheses distinguish direction, expected fractional return, horizon, causal chain, confidence, assumptions, contradictions and invalidation. Structured follow-ups use configured public sources; unresolved questions wait for new evidence and resume when it arrives. “Sources exhausted” is retained as unresolved rather than invented completion. A favorable development does not itself authorize a trade or assure a profit.

The controller computes event-study alpha/beta from pre-event observations, abnormal return R_asset-(alpha+beta*R_market), cumulative abnormal returns, response distributions, sample sizes/uncertainty, favorable/unfavorable excursions, abnormal volume and reversals. Missing samples remain explicitly insufficient. Qwen interprets these calculations; proximity alone is not a causal proof.

Forecasts mature only when a later available market observation reaches the specified horizon. Observed fractional error, direction accuracy and Brier score are stored separately from simulated execution. Persistent lessons retain supporting/contradictory examples and unresolved causal questions. Accuracy is grouped by issuer, industry, event classification, horizon and conditions. This is episodic evidence retrieval, not training of model weights.

## Operation and monitoring

The existing paper service performs acquisition/research overnight, on weekends and holidays without manufacturing trading sessions. During the session it handles the portfolio first; research follows that pass. After close it continues evidence acquisition and evaluation. Reinstall/update the service with scripts/install-paper-service.sh.

Read-only endpoints: /api/v1/intelligence/status, /graph, /evidence/{chunk_id}, /original/{version}. Evidence and graph endpoints accept an asof timestamp so later revisions are inaccessible to earlier decisions. The dashboard derives counts, source health, coverage, queued questions, forecasts and observed accuracy from these records.

Historical replay uses a separate paired intelligence database by default and never fetches future news into an earlier decision. Original availability timestamps continue to apply; importing a later revision does not make it contemporaneous with its old publication date.

## Validation

The suite includes acquisition parsing, original/version/chunk preservation, source outages/weekend scheduling, quoted-reference validation, restart recovery, future/revision exclusion, event-study arithmetic, unobserved forecast retention, grouped accuracy and a complete acquisition-to-research-to-ranked-paper-decision-to-fill-to-observation-to-lesson retrieval fixture. Frontend types/build and existing paper/backtest regressions run on GitHub. The default Qwen acceptance workflow downloads actual market bars and scores a paper forecast against withheld observations. The synthetic research probe remains available as an isolated diagnostic through app.intelligence.probe.

Research generation sends its typed JSON schema through Ollama's native format field. The controller still validates exact source quotations, available times, registered securities and semantic consistency; a structurally valid answer can still be rejected. Existing callers retain their original JSON/prose defaults and inference controls.

Treasury acquisition reads the department's official press-release index and queues its article links; it does not assume an RSS endpoint. HTTP content negotiation accepts standard XML media types. Updating a configured source replaces its adapter definition and marks obsolete pending poll jobs superseded while retaining their history.


The FRED adapter archives recent numeric observations. ALFRED queries observation values with their vintage intervals, queues further pages, and retains the raw provider payload and the acquisition timestamp. The observation date and vintage dates describe the economic record; the archive's availability timestamp controls point-in-time retrieval. Unconfigured optional sources leave other acquisition running.

Ranked opportunities record their technical, catalyst, historical-response, and liquidity contributions. Historical scores use bounded event-study samples and shrink small samples toward zero. The dashboard reports counts and recent coverage by symbol, interval, and session, plus daily accuracy, ranked evidence links and paper activity.
