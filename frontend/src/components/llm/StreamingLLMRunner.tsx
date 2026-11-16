'use client'

import React, { useState, useRef, useCallback, useEffect } from 'react'
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Progress } from '@/components/ui/progress'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Loader2, Play, Square, TrendingUp, BarChart3, Zap } from 'lucide-react'
import { BacktestChart } from '@/components/charts/BacktestChart'
import BacktestResultsWithAnalysis from '@/components/analysis/BacktestResultsWithAnalysis'

interface StreamMessage {
  type: 'start' | 'progress' | 'trading_progress' | 'result' | 'complete' | 'error'
  message?: string
  step?: string
  day?: number
  total_days?: number
  progress?: number
  event_type?: string
  data?: any
  // May be at the root level performance_metrics
  performance_metrics?: {
    total_return: number
    win_rate: number
    max_drawdown: number
    total_trades: number
    total_value: number
    cash: number
    position_value: number
  }
  // May be at the root level pnl_status
  pnl_status?: {
    unrealized_pnl?: number
    unrealized_pnl_pct?: number
    holding_days?: number
    shares?: number
    risk_level?: string
    cash_remaining?: number
    total_value?: number
  }
  extra_data?: {
    pnl_status?: {
      unrealized_pnl?: number
      unrealized_pnl_pct?: number
      holding_days?: number
      shares?: number
      risk_level?: string
      cash_remaining?: number
      total_value?: number
    }
    performance_metrics?: {
      total_return: number
      win_rate: number
      max_drawdown: number
      total_trades: number
      total_value: number
      cash: number
      position_value: number
    }
  }
}

interface DynamicPerformance {
  total_return: number     // Total return (Based on total value) 
  win_rate: number         // Win rate (0-1)
  max_drawdown: number     // Maximum drawdown (0-1)
  total_trades: number     // Number of completed trades (Meaningful) 
  total_realized_pnl?: number      // Cumulative realized P&L
  cumulative_trade_return_rate?: number  // Cumulative trade return
  // Can be added later: 
  // avg_trade_return?: number    // Average return per trade
  // profit_loss_ratio?: number   // Profit/loss ratio
  // max_single_loss?: number     // Largest single-trade loss
}

interface PnLStatus {
  unrealized_pnl?: number
  unrealized_pnl_pct?: number
  holding_days?: number
  shares?: number
  risk_level?: string
  cash_remaining?: number
  total_value?: number
}

interface BacktestResult {
  trades: any[]
  performance: any
  stock_data: any[]
  signals: any[]
  llm_decisions: any[]
  statistics: {
    total_trades: number
    win_rate: number
    total_return: number
    max_drawdown: number
    final_value?: number
    total_realized_pnl?: number
    cumulative_trade_return_rate?: number
  }
}

export default function StreamingLLMRunner() {
  const [symbol, setSymbol] = useState('AAPL')
  const [period, setPeriod] = useState('1y')
  const [initialCapital] = useState(100000000) // Set 1 hundred million as unlimited capital, Does not affect pure trading P&L calculations
  
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(0)
  const [currentStep, setCurrentStep] = useState('')
  const [messages, setMessages] = useState<string[]>([])
  const [result, setResult] = useState<BacktestResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isStarting, setIsStarting] = useState(false) // New: Prevent duplicate clicks
  const [currentRunId, setCurrentRunId] = useState<string | null>(null) // New: Track the current backtest’s unique identifier
  
  // Dynamic performance state
  const [dynamicPerformance, setDynamicPerformance] = useState<DynamicPerformance>({
    total_return: 0,
    win_rate: 0,
    max_drawdown: 0,
    total_trades: 0
  })
  
  // P&L State
  const [pnlStatus, setPnlStatus] = useState<PnLStatus | null>(null)
  
  // Real-time signal collection
  const [realTimeSignals, setRealTimeSignals] = useState<any[]>([])
  const [realTimeLLMDecisions, setRealTimeLLMDecisions] = useState<any[]>([])
  const [realTimeStockData, setRealTimeStockData] = useState<any[]>([])
  
  const eventSourceRef = useRef<EventSource | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const sessionIdRef = useRef<string | null>(null) // Add session ID

  // Cleanup function - Ensure EventSource Close correctly
  const cleanupEventSource = useCallback(() => {
    if (eventSourceRef.current) {
      console.log('Clean up EventSource Connection')
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    sessionIdRef.current = null
  }, [])

  // Clean up on component unmount
  useEffect(() => {
    return () => {
      cleanupEventSource()
    }
  }, [cleanupEventSource])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const addMessage = useCallback((message: string) => {
    setMessages(prev => [...prev, message])
    setTimeout(scrollToBottom, 100)
  }, [])

  const startStreaming = async () => {
    // Prevent duplicate clicks
    if (isRunning || isStarting) {
      console.log('Backtest already in progress, Ignore duplicate request')
      return
    }

    // Generate unique session ID and runId
    const sessionId = `session-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    const runId = `run-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    sessionIdRef.current = sessionId
    setCurrentRunId(runId)
    console.log('Start a new streaming backtest, Session ID:', sessionId, 'Run ID:', runId)
    
    setIsStarting(true)
    
    // Clean up previous connection first
    cleanupEventSource()

    setIsRunning(true)
    setProgress(0)
    setCurrentStep('')
    setMessages([])
    setResult(null)
    setError(null)
    setPnlStatus(null) // Reset P&L State
    
    // Reset real-time signal data
    setRealTimeSignals([])
    setRealTimeLLMDecisions([])
    setRealTimeStockData([])

    const params = new URLSearchParams({
      symbol,
      period,
      session_id: sessionId, // Add session ID
    })

    const url = `http://localhost:8000/api/v1/llm-stream/llm-backtest-stream?${params}`
    
    try {
      console.log('Create new EventSource:', url)
      eventSourceRef.current = new EventSource(url)
      
      eventSourceRef.current.onopen = () => {
        console.log('EventSource Connection established')
        setIsStarting(false)
      }
      
      eventSourceRef.current.onmessage = (event) => {
        try {
          const data: StreamMessage = JSON.parse(event.data)
          console.log('Streaming data received:', data.type, data.event_type, data.message)
          
          // Debug: Check performance_update Event
          if (data.event_type === 'performance_update') {
            console.log('Performance Update Detailed data:', {
              performance_metrics: data.performance_metrics,
              extra_data: data.extra_data,
              message: data.message
            })
          }
          
          switch (data.type) {
            case 'start':
              addMessage(data.message || 'Start backtest...')
              break
              
            case 'progress':
              setCurrentStep(data.message || '')
              // Show only important progress messages, Filter internal processing messages
              const progressMessage = data.message || ''
              if (!progressMessage.includes('Fetching') && 
                  !progressMessage.includes('Successfully retrieved') && 
                  !progressMessage.includes('Initialize') &&
                  !progressMessage.includes('Start execution') &&
                  !progressMessage.includes('Analyze backtest results')) {
                addMessage(progressMessage)
              }
              break
              
              case 'trading_progress':
                if (data.total_days && data.day) {
                  const progressPercent = (data.day / data.total_days) * 100
                  setProgress(progressPercent)
                  
                  // Handle consistently P&L Data updates - Update before all event types
                  let pnlData = data.extra_data?.pnl_status || data.pnl_status
                  if (pnlData) {
                    console.log('Update P&L State:', {
                      event_type: data.event_type,
                      holding_days: pnlData.holding_days,
                      unrealized_pnl: pnlData.unrealized_pnl,
                      shares: pnlData.shares,
                      full_data: pnlData
                    })
                    setPnlStatus(pnlData as PnLStatus)
                  }
                  
                  if (data.event_type === 'llm_decision') {
                    // Retain complete LLM decision content, for subsequent optimization analysis
                    const message = data.message || ''
                    addMessage(`🤖 ${message}`)
                    
                    // Collect LLM Decision data
                    if (data.extra_data) {
                      const llmDecision = {
                        date: new Date().toISOString(),
                        day: data.day,
                        decision: {
                          action: 'THINK', // LLM reasoning without a buy or sell signal
                          confidence: 0.8,
                          reason: message
                        },
                        price: (data.extra_data as any).current_price || 0,
                        timestamp: new Date().toISOString()
                      }
                      setRealTimeLLMDecisions(prev => [...prev, llmDecision])
                    }
                  } else if (data.event_type === 'signal_generated') {
                    // Improve signal generation display
                    const message = data.message || ''
                    const signalMatch = message.match(/(BUY|SELL).*?Confidence: ([\d.]+)/)
                    if (signalMatch) {
                      const signal = signalMatch[1]
                      const confidence = signalMatch[2]
                      const icon = signal === 'BUY' ? '🚀' : '📤'
                      addMessage(`${icon} Execute ${signal} Signal (Confidence: ${confidence})`)
                      
                      // Collect trading signal data
                      const tradingSignal = {
                        timestamp: new Date().toISOString(),
                        signal_type: signal,
                        price: (data.extra_data as any)?.current_price || 0,
                        confidence: parseFloat(confidence),
                        reason: message
                      }
                      setRealTimeSignals(prev => [...prev, tradingSignal])
                    } else {
                      addMessage(`📈 ${message}`)
                    }
                    
                    // Update performance data silently, Do not repeat messages (P&L Data already updated above) 
                    let signalMetrics = data.extra_data?.performance_metrics || (data as any).performance_metrics
                    let strategyStats = (data as any).strategy_statistics || (data.extra_data as any)?.strategy_statistics
                    
                    if (signalMetrics) {
                      setDynamicPerformance({
                        total_return: signalMetrics.total_return || 0,
                        win_rate: strategyStats?.strategy_win_rate || signalMetrics.win_rate || 0,
                        max_drawdown: signalMetrics.max_drawdown || 0,
                        total_trades: strategyStats?.total_trades || signalMetrics.total_trades || 0,
                        total_realized_pnl: strategyStats?.total_realized_pnl || signalMetrics.total_realized_pnl || 0,
                        cumulative_trade_return_rate: strategyStats?.cumulative_trade_return_rate || signalMetrics.cumulative_trade_return_rate || 0
                      })
                    }
                  } else if (data.event_type === 'llm_skipped') {
                    // Skip unimportant messages, Reduce log noise
                    // addMessage(`⏭️ ${data.message}`)
                  } else if (data.event_type === 'entry_point') {
                    addMessage(`🚀 ${data.message}`)
                  } else if (data.event_type === 'exit_point') {
                    addMessage(`📤 ${data.message}`)
                  } else if (data.event_type === 'performance_update') {
                    // Improve performance update logic, Avoid duplicate display
                    let metrics = data.extra_data?.performance_metrics || (data as any).performance_metrics
                    let strategyStats = (data as any).strategy_statistics || (data.extra_data as any)?.strategy_statistics
                    
                    if (metrics) {
                      const newTradeCount = strategyStats?.total_trades || metrics.total_trades || 0
                      const newReturn = metrics.total_return || 0
                      const newWinRate = strategyStats?.strategy_win_rate || metrics.win_rate || 0
                      
                      const prevTradeCount = dynamicPerformance.total_trades
                      const prevReturn = dynamicPerformance.total_return
                      
                      // Show trade completion only when the trade count actually increases
                      if (newTradeCount > prevTradeCount && newTradeCount > 0) {
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`💰 Trade completed | Total return: ${returnText}% | Win rate: ${winRateText}% | Completed trades: ${newTradeCount}`)
                      } else if (newTradeCount === 0 && prevTradeCount === 0 && Math.abs(newReturn - prevReturn) > 0.05) {
                        // Only when return changes substantially without a trade, show performance update (Avoid meaningless 0%Update) 
                        const returnText = (newReturn * 100).toFixed(2)
                        const winRateText = (newWinRate * 100).toFixed(1)
                        addMessage(`📊 Performance update | Total return: ${returnText}% | Win rate: ${winRateText}%`)
                      }
                      
                      setDynamicPerformance({
                        total_return: newReturn,
                        win_rate: newWinRate,
                        max_drawdown: metrics.max_drawdown || 0,
                        total_trades: newTradeCount,
                        total_realized_pnl: strategyStats?.total_realized_pnl || metrics.total_realized_pnl || 0,
                        cumulative_trade_return_rate: strategyStats?.cumulative_trade_return_rate || metrics.cumulative_trade_return_rate || 0
                      })
                    }
                    
                    // P&L State already updated above, Do not update again here
                  } else {
                    // Filter system messages, Show only important content
                    const message = data.message || ''
                    if (!message.includes('Processing progress') && 
                        !message.includes('Start LLM analysis') && 
                        message.trim() !== '') {
                      addMessage(message)
                    }
                  }
                }
                break
                
              case 'result':
              setResult(data.data)
              
              // Set complete stock data for charts
              if (data.data.stock_data) {
                setRealTimeStockData(data.data.stock_data)
              }
              
              // Update final performance Data, Prefer statistics strategy statistics
              const finalStrategyStats = data.data.strategy_statistics || {}
              const finalPerformance = data.data.performance || {}
              const finalStatistics = data.data.statistics || {}
              
              setDynamicPerformance({
                total_return: finalStatistics.total_return || finalPerformance.total_return || 0,
                win_rate: finalStatistics.win_rate / 100 || finalStrategyStats.strategy_win_rate || finalPerformance.win_rate || 0, // Convert percentages to decimals
                max_drawdown: finalStatistics.max_drawdown || finalPerformance.max_drawdown || 0,
                total_trades: finalStatistics.total_trades || finalStrategyStats.total_trades || 0,
                total_realized_pnl: finalStatistics.total_realized_pnl || finalStrategyStats.total_realized_pnl || 0,
                cumulative_trade_return_rate: finalStatistics.total_return / 100 || finalStrategyStats.cumulative_trade_return_rate || 0 // Use total return as cumulative trade return
              })
              
              addMessage('✅ Backtest completed. Generating charts...')
              break
              
            case 'complete':
              // Show only completion message, Do not display potentially inaccurate summary data
              addMessage('🎉 Backtest completed! See the cards below for accurate statistics')
              addMessage(data.message || 'All processing completed!')
              setIsRunning(false)
              cleanupEventSource()
              break
              
            case 'error':
              setError(data.message || 'An unknown error occurred')
              addMessage(`❌ Error: ${data.message}`)
              setIsRunning(false)
              cleanupEventSource()
              break
          }
        } catch (err) {
          console.error('Error parsing streaming data:', err)
        }
      }
      
      eventSourceRef.current.onerror = (event) => {
        console.error('EventSource Error:', event)
        setError('Connection interrupted or server error')
        setIsRunning(false)
        setIsStarting(false)
        cleanupEventSource()
      }
      
    } catch (err) {
      console.error('Error starting stream:', err)
      setError('Unable to start streaming backtest')
      setIsRunning(false)
      setIsStarting(false)
    }
  }

  const stopStreaming = () => {
    console.log('Manually stop stream')
    cleanupEventSource()
    setIsRunning(false)
    setIsStarting(false)
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="text-center">
        <h1 className="text-3xl font-bold gradient-text mb-2">
          🚀 Streaming LLM strategy backtest
        </h1>
        <p className="text-gray-600">Watch the AI trading strategy make decisions in real time.</p>
      </div>

      {/* Parameter settings */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5" />
            Backtest parameter settings
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div>
              <Label htmlFor="symbol">Stock symbol</Label>
              <Input
                id="symbol"
                value={symbol}
                onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                placeholder="For example: AAPL"
                disabled={isRunning}
              />
            </div>
            
            <div>
              <Label htmlFor="period">Backtest period</Label>
              <Select value={period} onValueChange={setPeriod} disabled={isRunning}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="6mo">6 months</SelectItem>
                  <SelectItem value="1y">1 year</SelectItem>
                  <SelectItem value="2y">2 years</SelectItem>
                  <SelectItem value="5y">5 years</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>
          
          <div className="bg-blue-50 p-4 rounded-lg mb-4">
            <div className="text-sm text-blue-800">
              <p className="font-medium">💰 Capital mode: Unlimited capital</p>
              <p className="text-xs mt-1">The system uses unlimited-capital mode. P&L calculations use actual transaction costs rather than the initial capital setting.</p>
            </div>
          </div>
          
          <div className="flex gap-2 mt-4">
            <Button 
              onClick={startStreaming} 
              disabled={isRunning || isStarting} 
              className="flex-1"
            >
              {(isRunning || isStarting) ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  {isStarting ? 'Starting...' : 'Backtest in progress...'}
                </>
              ) : (
                <>
                  <Play className="mr-2 h-4 w-4" />
                  Start streaming backtest
                </>
              )}
            </Button>
            
            {(isRunning || isStarting) && (
              <Button onClick={stopStreaming} variant="destructive">
                <Square className="mr-2 h-4 w-4" />
                Stop
              </Button>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Progress display */}
      {isRunning && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Zap className="h-5 w-5" />
              Live progress and performance
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              <div>
                <div className="flex justify-between text-sm mb-2">
                  <span>Processing progress</span>
                  <span>{progress.toFixed(1)}%</span>
                </div>
                <Progress value={progress} className="w-full" />
              </div>
              
              {/* Dynamic performance metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="text-center p-2 bg-green-50 rounded">
                  <div className="text-lg font-bold text-green-600">
                    {dynamicPerformance.total_trades}
                  </div>
                  <div className="text-xs text-gray-600">Completed trades</div>
                </div>
                <div className="text-center p-2 bg-blue-50 rounded">
                  <div className="text-lg font-bold text-blue-600">
                    {(dynamicPerformance.win_rate * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-600">Strategy win rate</div>
                </div>
                <div className="text-center p-2 bg-purple-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.total_realized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    ${(dynamicPerformance.total_realized_pnl ?? 0).toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-600">Cumulative realized P&L</div>
                </div>
                <div className="text-center p-2 bg-orange-50 rounded">
                  <div className={`text-lg font-bold ${(dynamicPerformance.cumulative_trade_return_rate ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                    {((dynamicPerformance.cumulative_trade_return_rate ?? 0) * 100).toFixed(2)}%
                  </div>
                  <div className="text-xs text-gray-600">Cumulative trade return</div>
                </div>
              </div>
              
              {/* P&L State display */}
              {pnlStatus && (
                <div className="border rounded-lg p-4 bg-gradient-to-r from-green-50 to-blue-50">
                  <div className="text-sm font-semibold mb-3 flex items-center gap-2">
                    <TrendingUp className="h-4 w-4" />
                    Current trading state
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        ${(pnlStatus.unrealized_pnl ?? 0).toFixed(2)}
                      </div>
                      <div className="text-xs text-gray-600">Unrealized P&L</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-xl font-bold ${(pnlStatus.unrealized_pnl_pct ?? 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {(pnlStatus.unrealized_pnl_pct ?? 0).toFixed(2)}%
                      </div>
                      <div className="text-xs text-gray-600">Current trade return</div>
                    </div>
                    <div className="text-center">
                      <div className="text-xl font-bold text-blue-600">
                        {pnlStatus.shares ? `${(pnlStatus.shares / 1000).toFixed(1)}k shares` : 'No position'}
                      </div>
                      <div className="text-xs text-gray-600">Shares held</div>
                    </div>
                  </div>
                  <div className="mt-3 text-xs text-gray-500 text-center">
                    Risk level: <span className={`font-semibold ${
                      pnlStatus.risk_level === 'high' ? 'text-red-600' : 
                      pnlStatus.risk_level === 'medium' ? 'text-yellow-600' : 'text-green-600'
                    }`}>{pnlStatus.risk_level ?? 'normal'}</span>
                  </div>
                </div>
              )}
              
              {currentStep && (
                <div className="p-3 bg-blue-50 rounded-lg">
                  <p className="text-sm text-blue-800">{currentStep}</p>
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Live logs */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5" />
            Live decision logs
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="h-96 overflow-y-auto bg-gray-50 p-4 rounded-lg space-y-2">
            {messages.map((message, index) => {
              // Style based on message type
              let messageClass = "text-sm p-3 rounded-md leading-relaxed"
              
              if (message.includes('🤖') && message.includes('LLM Decision')) {
                // LLM Decision message - Special styling, More room for complete content
                messageClass += " bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 text-blue-900"
              } else if (message.includes('🟢') || message.includes('🚀')) {
                // Buy-related messages
                messageClass += " bg-green-100 border-l-4 border-green-500 text-green-800"
              } else if (message.includes('🔴') || message.includes('📤')) {
                // Sell-related messages  
                messageClass += " bg-red-100 border-l-4 border-red-500 text-red-800"
              } else if (message.includes('🟡')) {
                // Hold-related messages
                messageClass += " bg-yellow-100 border-l-4 border-yellow-500 text-yellow-800"
              } else if (message.includes('💰')) {
                // Performance update messages
                messageClass += " bg-blue-100 border-l-4 border-blue-500 text-blue-800 font-semibold"
              } else if (message.includes('✅') || message.includes('Completed')) {
                // Completion message
                messageClass += " bg-purple-100 border-l-4 border-purple-500 text-purple-800"
              } else {
                // General message
                messageClass += " bg-white border-l-4 border-gray-300 text-gray-700"
              }
              
              return (
                <div key={index} className={messageClass}>
                  <div className="flex items-start gap-2">
                    <span className="text-xs text-gray-500 min-w-fit">
                      [{new Date().toLocaleTimeString('en-US')}]
                    </span>
                    <span className="flex-1 whitespace-pre-wrap break-words">{message}</span>
                  </div>
                </div>
              )
            })}
            <div ref={messagesEndRef} />
          </div>
        </CardContent>
      </Card>

      {/* Error display */}
      {error && (
        <Alert variant="destructive">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {/* Results display */}
      {result && currentRunId && (
        <BacktestResultsWithAnalysis
          backtestResult={result}
          runId={currentRunId}
        />
      )}
    </div>
  )
}
