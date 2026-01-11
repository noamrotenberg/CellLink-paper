import pandas as pd
# Remove build metadata from pandas.__version__ - causes issue with sssom
pd.__version__ = pd.__version__.split("+")[0]

from oaklib import get_adapter

# import asyncio

# this script based on Chris Mungall's agentic AI tutorial https://github.com/ai4curation/agent-tutorial




adapter = get_adapter("ols:cl")
# print("Agent query:", term)
results = adapter.basic_search("MSC")
print("results:", results)
print("adapter.labels(results):", adapter.labels(results))
labels = list(adapter.labels(results))

# remove any terms imported from other ontologies
print("labels:", labels)
labels = list(filter(lambda label: (label[0] is not None) and (label[0].startswith("CL:")), labels))
labels = labels[:20] # limit to first 20 results
if len(labels) == 0:
    labels = "No results were found. Try another query."
    # without adding this text, the model sometimes repeatedly makes the same call
