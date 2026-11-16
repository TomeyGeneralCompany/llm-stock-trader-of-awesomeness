'use client'
import { useQuery } from '@tanstack/react-query'

interface Status {
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
  const data = query.data
  return (
    <section className="max-w-7xl mx-auto p-6 space-y-4" aria-label="Market intelligence">
      <h2 className="text-2xl font-semibold">Market intelligence</h2>
      {query.error && <p role="alert">{query.error.message}</p>}
      {!data && !query.error && <p>Loading evidence archive…</p>}
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
