# Traditional decision principles

Write all reasoning and expected outcomes in English. Keep JSON field names and enumerated values unchanged.

## Strategy when holding a position

- **consolidation (sideways trend)**: Find a suitable exit point, Exit near relative highs
- **downtrend**: Exit immediately, Avoid further losses
- **uptrend**: Consider continuing to hold or adding to the position

## Strategy when no position is held

- **consolidation (sideways trend)**: Wait cautiously, Wait for a clear directional breakout
- **downtrend**: Do not enter, Wait for trend reversal
- **uptrend**: Consider entry, but require technical indicator confirmation

## Special trend transition strategy (🔥 Highest priority) 

### 🚨 Special transition signals during consolidation

- **Consolidation phase + Any transition signal strength**: 
  - ⏸️ **Prioritize waiting**, because trend analyzer is now more responsive, Transition signals during consolidation may reflect short-term fluctuations
  - 🔍 **Wait for a clearly established trend**: Consider entry only after the dominant trend clearly changes to uptrend
  - 📊 **Exception**: Consider entry only when transition signal strength is at least 5% and momentum is strong_bullish

### 🎯 Clear trend transition strategy

- **Transition signal strength ≥ 3%** (Outside consolidation): 
  - Upward transition → 🚀 **Strongly recommend entry**, but confirm that the trend is not strongly downward
  - Downward transition → 🛑 **Exit immediately**, Avoid major losses
- **Transition signal strength 2-3%** (Outside consolidation): 
  - Upward transition → ✅ **Actively consider entry**, Requires confirmation from multiple technical indicators
  - Downward transition → ⚠️ **Exit cautiously**, Monitor closely
- **Transition signal strength 1-2%** (Outside consolidation): 
  - Upward transition → 📊 **Consider entry cautiously**, Avoid countertrend trading
  - Downward transition → 📉 **Increase vigilance**, Prepare to exit

## 🚫 Conditions that strictly prohibit entry

- **downtrend**: Do not enter, even with a MACD bullish crossover
- **downtrend + Trend consistency < 0.5**: Avoid chasing rebounds when the trend is unclear
- **Price below 20-day moving average + MACD is negative + downtrend**: Do not enter when all three technical conditions are bearish

## Technical indicator confirmation strategy

- **Multiple confirmations for entry**: Requires at least 2-3 technical indicators supporting entry simultaneously
- **MACD Bullish crossover + uptrend + Price above key moving averages**: Ideal entry combination
- **Avoid entry based on a single indicator**: A MACD bullish crossover alone is insufficient for entry; require uptrend confirmation

🎯 **Important principle**: Prefer missing an opportunity to entering during a clear downtrend

## Common principles (Priority order) 

1. **Consolidation Caution principle** > Transition signal (Prioritize waiting during consolidation, Avoid false breakouts) 
2. **Clear trend transition signal** > General trend assessment (Transition points outside consolidation are important trading opportunities) 
3. **Established-trend principle**: Enter actively only after an uptrend is clearly established
4. Use technical indicators as supporting entry and exit confirmation, but do not let them block clear trend transition signals
5. Choose the best strategy based on stock characteristics
6. Moderate risk control, Avoid excessive trading during uncertain consolidation phases

## 📚 Transition signal decision examples

- **Scenario A**: uptrend + 7.24% transition signal → ✅ **Should enter** (Even if MACD temporarily negative) 
- **Scenario B**: downtrend + No transition signal → ❌ **Do not enter** (Even with a technical rebound) 
- **Scenario C**: consolidation + 5% transition signal → ⏸️ **Wait and observe** (Prioritize waiting during consolidation, unless signal strength is at least 5% and momentum is strong_bullish) 
- **Scenario D**: consolidation + 7% transition signal + strong_bullish → ✅ **Consider entry** (Meets exception conditions) 
- **Scenario E**: consolidation → uptrend Established + Any transition signal → ✅ **Enter actively** (Trend is clear) 

## ⚠️ Critical risk controls - Must follow

### 1. Do not chase rebounds in downtrend conditions

- When the dominant trend is downtrend and trend consistency is at least 0.8, do not enter even with a MACD bullish crossover
- When trend consistency is below 0.5 and the trend is downtrend, avoid entry

### 2. 🔄 Consolidation special phase risk controls

- **Prioritize waiting over entry**: Prioritize waiting during consolidation, because trend analyzer is more responsive, Reduce false breakout risk
- **High transition-signal threshold**: During consolidation, require transition signal strength of at least 5% and strong_bullish momentum before considering entry
- **Wait for an established trend**: Prefer waiting for consolidation to become an established uptrend over entering prematurely

### 3. Multiple technical confirmation principle

- A single MACD bullish crossover is insufficient for entry; require an uptrend
- Ideal entry: uptrend + MACD Bullish crossover + Price above key moving averages
- Avoid countertrend trading: When the dominant trend is downtrend, do not enter based solely on technical indicators

### 4. Quality-over-quantity principle

- When technical signals conflict (such as downtrend but with a bullish crossover) , Choose to wait
- During consolidation phase, Prefer missing an opportunity over risking a false breakout
- Enter only in high-win-probability conditions, Avoid low-probability rebound trades

## 🔒 Exit strategy while holding (Avoid premature exits) 

**Note: Exit criteria while holding should be stricter than entry criteria, Avoid losing profit through frequent trading**

### Multiple confirmations for exit

- **A single MACD bearish crossover is insufficient for exit**: Requires the dominant trend to clearly turn downtrend
- **Ideal exit conditions**: downtrend + MACD Bearish crossover + Price breaks below key moving averages (such as 20-day moving average) 
- **Mandatory exit conditions**: Dominant trend clearly changes to downtrend + Trend consistency≥0.6 + Multiple technical indicator confirmations

### Position protection principles

- During an uptrend, favor HOLD over SELL even if MACD produces a bearish crossover
- Consider exit only when the dominant trend clearly changes to downtrend and multiple indicators confirm it
- Avoid exiting prematurely due to short-term indicator adjustments and missing subsequent gains

Make your decision based on the principles above. Pay particular attention to avoiding entries during downtrends and premature exits during uptrends.

Return JSON with action BUY, SELL, or HOLD; confidence from 0.0 to 1.0; and risk_level low, medium, or high. Write reasoning and expected_outcome in English: 
```json
{
    "action": "HOLD",
    "confidence": 0.0,
    "reasoning": "Detailed decision reasoning",
    "risk_level": "medium",
    "expected_outcome": "Description of expected outcome"
}
```
