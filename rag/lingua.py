from llmlingua import PromptCompressor

llm_lingua = PromptCompressor(device_map="cpu")
prompt="""
Lambda calculus may be untyped or typed. In typed lambda calculus, functions can beapplied only if they are capable of accepting the given input's "type" of data. Typed lambdacalculi are weaker than the untyped lambda calculus, which is the primary subject of thisarticle, in the sense that typed lambda calculi can express less than the untyped calculuscan, On the other hand, typed lambda calculi allow more things to be proven. For examplein the simply typed lambda calculus it is a theorem that every evaluation strategyterminates for every simply typed lambda-term, whereas evaluation ofuntyped lambda.terms need not terminate, One reason there are many different typed lambda calculi hasbeen the desire to do more (of what the untyped calculus can do) without giving up onbeing able to prove strong theorems about the calculus.
"""
compressed_prompt = llm_lingua.compress_prompt(prompt, instruction="", question="")

print(compressed_prompt)
# > {'compressed_prompt': 'Question: Sam bought a dozen boxes, each with 30 highlighter pens inside, for $10 each box. He reanged five of boxes into packages of sixlters each and sold them $3 per. He sold the rest theters separately at the of three pens $2. How much did make in total, dollars?\nLets think step step\nSam bought 1 boxes x00 oflters.\nHe bought 12 * 300ters in total\nSam then took 5 boxes 6ters0ters.\nHe sold these boxes for 5 *5\nAfterelling these  boxes there were 3030 highlighters remaining.\nThese form 330 / 3 = 110 groups of three pens.\nHe sold each of these groups for $2 each, so made 110 * 2 = $220 from them.\nIn total, then, he earned $220 + $15 = $235.\nSince his original cost was $120, he earned $235 - $120 = $115 in profit.\nThe answer is 115',
#  'origin_tokens': 2365,
#  'compressed_tokens': 211,
#  'ratio': '11.2x',
#  'saving': ', Saving $0.1 in GPT-4.'}

## Or use the phi-2 model,
#llm_lingua = PromptCompressor("Qwen/Qwen2.5-0.5B-Instruct",device_map="cpu")

## Or use the quantation model, like TheBloke/Llama-2-7b-Chat-GPTQ, only need <8GB GPU memory.
## Before that, you need to pip install optimum auto-gptq
# llm_lingua = PromptCompressor("TheBloke/Llama-2-7b-Chat-GPTQ", model_config={"revision": "main"})