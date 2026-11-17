'use client'

import { useQuery } from '@tanstack/react-query'

type PaperReport = {
  mode: string
  challenge: boolean
  equity: number
  buying_power_model: string
  forecasts: number
  scored: number
  state: { day: string | null; cash: number; unsettled: number; closed: boolean; goal_reached_at: string | null; last_poll: string | null; holdings: Record<string, { shares: number; basis: number }> }
  sessions: {
    day: string
    equity: number
    complete: boolean
    achieved: boolean | null
    coverage?: { observed_bars: number; expected_bars: number; late_or_unverified_bar_ends: string[]; heartbeat_gaps: unknown[] }
  }[]
  analysis_timing?: {
    completed: number
    current_when_finished: number
    queued: number
    rejected: number
    recent: { symbol: string; finished: string; disposition?: string; reason?: string;
      feed_lag_seconds?: number | null; queue_seconds?: number; inference_seconds?: number }[]
  }
  events: Record<string, number>
  latest_decisions: { symbol: string; bar: string; decision: { action: string; reasoning: string; confidence: number } }[]
}
const seconds = (n?: number | null) => n == null ? '—' : n.toFixed(1) + ' s'
const money = (n: number) => '$' + n.toFixed(2)

export default function PaperTraderStatus() {
  const query = useQuery<PaperReport | null>({
    queryKey: ['paper-status'],
    queryFn: async () => {
      const r = await fetch('/api/v1/paper/status', { cache: 'no-store' })
      if (r.status === 404) return null
      if (!r.ok) throw new Error('Paper status unavailable')
      return r.json()
    },
    refetchInterval: 15000,
  })
  const d = query.data
  return (
    <section className="mx-auto max-w-7xl px-6 py-4" aria-label="Autonomous paper trader">
      <div className="rounded-lg border p-4">
        <h2 className="font-semibold text-lg">Autonomous paper trader</h2>
        {query.isLoading ? <p>Reading account…</p> : query.isError ? <p>Paper status unavailable.</p> : !d ? (
          <p className="text-sm">No local paper account. Start the paper-trader service to monitor its simulated performance here.</p>
        ) : (
          <>
            <p className="text-sm text-muted-foreground">{d.mode} · {d.buying_power_model} · no broker execution</p>
            <div className="grid grid-cols-2 gap-3 my-4 md:grid-cols-4">
              <div><p className="text-xs">Net marked equity</p><strong>{money(d.equity)}</strong></div>
              <div><p className="text-xs">Available ledger cash</p><strong>{money(d.state.cash)}</strong></div>
              <div><p className="text-xs">Unsettled</p><strong>{money(d.state.unsettled)}</strong></div>
              <div><p className="text-xs">Forecasts scored</p><strong>{d.scored} / {d.forecasts}</strong></div>
            </div>
            {d.challenge && <p className="text-sm font-medium">Daily objective: $1 → $100 · {d.state.goal_reached_at ? 'Achieved by realized trades' : 'Not achieved yet'}</p>}
            <p className="text-xs text-muted-foreground">Session: {d.state.day || 'Waiting for market data'} · {d.state.closed ? 'Closed' : 'Not closed'} · Last observation: {d.state.last_poll || 'None'}</p>
            <div className="grid gap-4 md:grid-cols-2 mt-3">
              <div><h3 className="font-medium">Holdings</h3>{Object.entries(d.state.holdings).map(([sym,p]) => <p className="text-sm" key={sym}>{sym}: {p.shares.toFixed(6)} shares · basis {money(p.basis)}</p>)}</div>
              <div><h3 className="font-medium">Latest decisions</h3>{d.latest_decisions.slice(0,4).map((x,i) => <p className="text-sm" key={i}>{x.symbol} · {x.decision.action} · {x.decision.reasoning}</p>)}</div>
            </div>
            {d.analysis_timing && d.analysis_timing.recent.length > 0 && (
              <details className="mt-3">
                <summary>Decision timing</summary>
                <p className="text-sm">{d.analysis_timing.current_when_finished} of {d.analysis_timing.completed} decisions finished while current. {d.analysis_timing.queued} orders queued, {d.analysis_timing.rejected} rejected.</p>
                {d.analysis_timing.recent.map((row, index) => <p key={index} className="text-sm">{row.symbol} · {row.disposition || 'Analysis error'} · feed {seconds(row.feed_lag_seconds)} · queue {seconds(row.queue_seconds)} · inference {seconds(row.inference_seconds)} {row.reason && '· ' + row.reason}</p>)}
              </details>
            )}
            <h3 className="font-medium mt-3">Session records</h3>
            {d.sessions.slice(-5).reverse().map(s => <p key={s.day} className="text-sm">
              {s.day} · {money(s.equity)} · {!s.coverage ? 'Coverage unverified' : !s.complete ? 'Incomplete data' : s.achieved ? 'Goal reached' : 'Closed'}
              {s.coverage && ' · ' + s.coverage.observed_bars + '/' + s.coverage.expected_bars + ' bars · ' + s.coverage.late_or_unverified_bar_ends.length + ' late · ' + s.coverage.heartbeat_gaps.length + ' heartbeat gaps'}
            </p>)}
          </>
        )}
      </div>
    </section>
  )
}
