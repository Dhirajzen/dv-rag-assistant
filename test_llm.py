"""Sanity check: is the Anthropic API key valid and the configured model reachable?"""
from rag import LLM_MODEL, get_llm

print(f"Model: {LLM_MODEL}")
print(get_llm().invoke("In one sentence, what is a protocol specification?").text)
