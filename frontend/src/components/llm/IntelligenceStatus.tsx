'use client'
import { useQuery } from '@tanstack/react-query'

interface PaperStatus {
  last_quote_received: string | null
  ranked_at: string | null
  ranked_opportunities: { symbol: string; score: number; bar: string; components?: Record<string, number>;
    historical_samples?: number; median_dollar_volume?: number | null; minutes_remaining?: number | null;
    catalysts: { evidence: { chunk_id: string }[] }[] }[]
  sessions: { day: string; complete: boolean; achieved: boolean; equity: number; holdings: Record<string, unknown>; cash: number; unsettled: number }[]
  latest_decisions: { symbol: string; bar: string; decision: {action: string; reasoning: string} }[]
  recent_events: { time: string; kind: string; detail: {symbol?: string; price?: number; reason?: string; fill_method?: string} }[]
}

interface Status {
  accuracy_over_time?: { day: string; n: number; mae: number; directional_accuracy: number; brier: number }[]
  coverage_details?: {symbol: string; kind: string; session?: string; status: string; earliest: string | null; latest: string | null}[]
  status: string
  lessons: { symbol: string; available: string; body: string }[]
  activity: { id: number; stamp: string; stage: string; body: string }[]
  counts: Record<string, number>
  sources: { id: string; last_success: string | null; failures: number; error: string | null }[]
  investigations: { id: string; question: string; status: string; steps: number; error: string | null }[]
  accuracy: { group: string; label: string; n: number; mae: number; directional_accuracy: number; brier: number }[]
  forecasts: { id: string; symbol: string; target: string; status: string; body: string }[]
  coverage: { status: string; count: number }[]
}
export default function IntelligenceStatus() {
  const query = useQuery<Status>({
    queryKey: ['market-intelligence'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/v1/intelligence/status')
      if (!response.ok) throw new Error('Unable to load saved intelligence status')
      return response.json()
    },
    refetchInterval: 15000,
  })
  const paper = useQuery<PaperStatus | null>({
    queryKey: ['paper-intelligence-paper-status'],
    queryFn: async () => {
      const response = await fetch('http://localhost:8000/api/v1/paper/status')
      if (response.status === 404) return null
      if (!response.ok) throw new Error('Unable to load the local paper account')
      return response.json()
    },
    refetchInterval: 15000,
  })
  const data = query.data
  return (
    <section className="max-w-7xl mx-auto p-6 space-y-4" aria-label="Market intelligence">
      <h2 className="text-2xl font-semibold">Market intelligence</h2>
      {query.error && <p role="alert">{query.error.message}</p>}
      {!data && !query.error && <p>Loading evidence archive…</p>}
      {paper.error && <p role="alert">{paper.error.message}</p>}
      {paper.data && <>
        <p className="text-sm">Last quote received {paper.data.last_quote_received || '— waiting for prices'}</p>
        <details className="border rounded p-3">
          <summary>Ranked opportunities</summary>
          <p className="text-sm">Updated {paper.data.ranked_at || 'after the first scan'}</p>
          {(paper.data.ranked_opportunities || []).map(item => <div className="my-3 text-sm" key={item.symbol}>
            <strong>{item.symbol}</strong> · score {item.score.toFixed(4)}
            <p>{Object.entries(item.components || {}).map(([name, score]) => name + ' ' + score.toFixed(4)).join(' · ')}</p>
            <p>{item.historical_samples || 0} comparable observations · median dollar volume {item.median_dollar_volume?.toLocaleString() ?? 'unavailable'} · minutes remaining {item.minutes_remaining?.toFixed(0) ?? 'unavailable'}</p>
            {item.catalysts.flatMap(catalyst => catalyst.evidence).map((evidence, index) => <a key={evidence.chunk_id + index} className="underline mr-3" href={'http://localhost:8000/api/v1/intelligence/evidence/' + encodeURIComponent(evidence.chunk_id) + '?asof=' + encodeURIComponent(paper.data?.ranked_at || item.bar)}>Evidence {index + 1}</a>)}
          </div>)}
        </details>
        <details className="border rounded p-3">
          <summary>Recent paper activity</summary>
          {paper.data.latest_decisions.map(row => <p className="my-2 text-sm" key={row.symbol + row.bar}><strong>{row.symbol} {row.decision.action}</strong> · {row.bar} · {row.decision.reasoning}</p>)}
          {paper.data.recent_events.filter(row => ['fill', 'unfilled', 'order_not_queued'].includes(row.kind)).map((row, index) => <p className="my-2 text-sm" key={index}><strong>{row.kind}</strong> · {row.time} · {row.detail.symbol} {row.detail.price ?? ''} {row.detail.fill_method || row.detail.reason || ''}</p>)}
          {paper.data.sessions.slice(-5).map(row => {
            const passed = row.complete && row.achieved && Object.keys(row.holdings).length === 0 && row.cash + row.unsettled >= 100
            return <p key={row.day} className="my-2 text-sm"><strong>{row.day}</strong> · {passed ? 'Goal reached' : row.complete ? 'Session complete' : 'Incomplete session'} · equity {row.equity.toLocaleString('en-US', {style: 'currency', currency: 'USD'})}</p>
          })}
        </details>
      </>}
      {data && <>
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
          {[
            ['Securities', data.counts.securities || 0],
            ['Documents', data.counts.document_versions || 0],
            ['Relationships', data.counts.entity_relationships || 0],
            ['Investigations', data.counts.research_tasks || 0],
            ['Forecasts', data.counts.market_forecasts || 0],
            ['Lessons', data.counts.learned_lessons || 0],
          ].map(([label, count]) => <div key={String(label)} className="border rounded p-3"><div>{label}</div><strong>{count}</strong></div>)}
        </div>
        <p className="text-sm">Acquisition and research continue outside trading hours. These findings inform the paper trader; research does not place orders.</p>
        <p className="text-sm">Coverage: {data.coverage.map(x => x.status + ': ' + x.count).join(' · ') || 'Waiting for the first acquisitions.'}</p>
        <details className="border rounded p-3">
          <summary>Sources and availability</summary>
          {data.sources.map(source => <div key={source.id} className="my-2 text-sm"><strong>{source.id}</strong>: {source.last_success ? 'Last success ' + source.last_success : 'Pending acquisition'}{source.error && <span className="text-amber-700"> — {source.error}</span>}</div>)}
        </details>
        <details className="border rounded p-3">
          <summary>Investigations and unanswered questions</summary>
          {data.investigations.map(task => <div key={task.id} className="my-3 text-sm"><strong>{task.status}</strong> · {task.steps} completed steps<p>{task.question}</p>{task.error && <p>{task.error}</p>}</div>)}
        </details>
        <details className="border rounded p-3">
          <summary>Forecasts and evidence</summary>
          {data.forecasts.map(forecast => {
            const body = JSON.parse(forecast.body) as { expected_return: number; confidence: number; invalidation: string; evidence: { chunk_id: string }[] }
            return <div key={forecast.id} className="my-3 text-sm"><strong>{forecast.symbol}</strong>: {(body.expected_return * 100).toFixed(2)}% forecast · confidence {body.confidence} · {forecast.status}<p>Horizon ends {forecast.target}. Invalidation: {body.invalidation}</p>{body.evidence.map((evidence, index) => <a key={index} className="underline mr-3" href={'http://localhost:8000/api/v1/intelligence/evidence/' + encodeURIComponent(evidence.chunk_id)} target="_blank" rel="noreferrer">Evidence {index + 1}</a>)}</div>
          })}
        </details>
        <details className="border rounded p-3">
          <summary>Saved lessons and recent work</summary>
          {data.lessons.map((lesson, index) => {
            const body = JSON.parse(lesson.body) as { actual_return?: number; absolute_error?: number; error_classes?: string[]; causal_interpretation?: string }
            return <div key={lesson.available + index} className="my-3 text-sm"><strong>{lesson.symbol}</strong> · {lesson.available}<p>{body.causal_interpretation || 'Evidence-backed outcome awaiting causal review.'}</p><p>{body.actual_return !== undefined && 'Observed return: ' + (body.actual_return * 100).toFixed(2) + '% · '}{body.error_classes?.join(', ') || 'No classified discrepancy'}</p></div>
          })}
          {!data.lessons.length && <p>Waiting for observed outcomes.</p>}
          {data.activity.slice(0, 12).map(event => <p key={event.id} className="text-sm my-1"><strong>{event.stage}</strong> · {event.stamp} · {event.body}</p>)}
        </details>
        <details className="border rounded p-3">
          <summary>Observed forecast accuracy</summary>
          <table className="w-full text-sm"><thead><tr><th>Group</th><th>Samples</th><th>MAE</th><th>Direction</th><th>Brier</th></tr></thead>
            <tbody>{data.accuracy.map(group => <tr key={group.group + group.label}><td>{group.group}: {group.label}</td><td>{group.n}</td><td>{group.mae.toFixed(4)}</td><td>{(group.directional_accuracy * 100).toFixed(1)}%</td><td>{group.brier.toFixed(4)}</td></tr>)}</tbody></table>
          {!data.accuracy.length && <p>No matured forecasts have observed outcomes yet.</p>}
        </details>
      </>}
    </section>
  )
}
