"""Native output structure; controller facts and execution rules remain authoritative."""


def paper_decision_schema(context):
    def obj(properties):
        return {"type":"object","properties":properties,"required":list(properties),"additionalProperties":False}
    def text():
        return {"type":"string","minLength":1}
    def enum(values):
        return {"type":"string","enum":list(values)}
    def array(items, minimum=0):
        return {"type":"array","items":items,"minItems":minimum}
    def number(low, high):
        return {"type":"number","minimum":low,"maximum":high}
    evidence, numeric = [], []
    import math
    for group in ("current","technical_facts","trend"):
        for key,value in context.get(group,{}).items():
            if value is not None and value != "unknown" and not isinstance(value,(dict,list)):
                name=group+"."+key
                if not isinstance(value,float) or math.isfinite(value):
                    evidence.append(name)
                    if group!="trend" and isinstance(value,(int,float)) and not isinstance(value,bool):
                        numeric.append(name)
    for key,value in context.items():
        if key.startswith("macd_") and value is not None and value!="unknown":
            evidence.append(key)
    if not evidence or not numeric:
        raise ValueError("Paper decisions require available controller evidence and numeric watch fields.")
    position="long" if context["shares"]>0 else "flat"
    phase=context["trend"].get("market_phase") or context["trend"].get("dominant_trend") or "unknown"
    observed=obj({"timeframe":text(),"position":enum([position]),"market_phase":enum([phase]),
                  "macd_cross":enum([context["macd_cross"]]),
                  "macd_alignment":enum([context.get("macd_alignment","unknown")]),
                  "evidence":array(text(),1)})
    forecast=obj({"scenario":text(),"timeframe":text(),"assumptions":array(text(),1),"invalidation":text()})
    plan=obj({"action":enum(["BUY","SELL","HOLD"]),"trigger":text(),"timeframe":text(),
              "required_position":enum(["flat","long","any"]),"prerequisite":text(),"reasoning":text()})
    risk=obj({"evidence":array(enum(evidence),1),"rationale":text(),"downside":text()})
    execution=obj({"mode":enum(["now","wait"]),"evidence":array(enum(evidence),1),"rationale":text()})
    explanation=obj({"observed_state":observed,"interpretation":text(),"forecast":forecast,
                     "conditional_actions":array(plan,1),"risk_assessment":risk,"execution":execution})
    watch=obj({"key":enum(numeric),"op":enum([">=","<="]),"threshold":number(-1e15,1e15),
               "required_position":enum(["flat","long","any"])})
    watches=array(watch)
    watches["maxItems"]=3
    return obj({"action":enum(["BUY","SELL","HOLD"] if position=="long" else ["BUY","HOLD"]),
                "confidence":number(0,1),"risk_level":enum(["low","medium","high"]),
                "reasoning":text(),"expected_outcome":text(),"explanation":explanation,
                "allocation_fraction":number(0,1),"expected_return":number(-1,100),
                "horizon_bars":{"type":"integer","minimum":1,"maximum":12},"watch_conditions":watches})
