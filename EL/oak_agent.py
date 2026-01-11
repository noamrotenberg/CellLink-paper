import pandas as pd
# Remove build metadata from pandas.__version__ - causes issue with sssom
pd.__version__ = pd.__version__.split("+")[0]

from typing import List, Tuple
from oaklib import get_adapter
import pydantic
import pydantic_ai
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import openai
import os
# import asyncio

# this script based on Chris Mungall's agentic AI tutorial https://github.com/ai4curation/agent-tutorial

def set_up_model(model_name):
    print("Setting up oak_agent.")
    
    azure_client = openai.AsyncAzureOpenAI(
        azure_endpoint=os.environ['endpoint_agent'],
        api_version="2025-03-01-preview",
        api_key=os.environ['api_key_agent'],
    )
    
    print("Succeeded to create azure_client.")
    
    model = OpenAIModel(
        model_name,
        provider=OpenAIProvider(openai_client=azure_client,
                                # api_type="azure"
                                )
    )
    
    oak_agent = pydantic_ai.Agent(  
        model,
        system_prompt="""
        You are an expert ontology curator. Use the Cell Ontology search tool at your 
        disposal find an identifier that best matches the user's input cell type.
        Not all inputs can be linked. Choose the best match if you cannot find exactly 
        what you're looking for after querying the ontology in a few different ways.
        """,
        tools=[search_CL],
        result_type=CL_identifier,
        model_settings={"do_sample": False}
        # note: even with do_sample=False, the agent does not appear to be deterministic
    )

    # print("oak_agent:", oak_agent)
    return oak_agent


class CL_identifier(pydantic.BaseModel):
    """
    A CL identifier has only a string of the form "CL:" followed by a string of numbers.
    """
    identifier: str


def search_CL(term: str) -> List[Tuple[str, str]]:
    """
    Search the Cell Ontology (CL) for a term.

    Note that search should take into account synonyms, but synonyms may be incomplete,
    so if you cannot find a concept of interest, try searching using related or synonymous
    terms.

    If you are searching for a composite term, try searching on the sub-terms to get a sense
    of the terminology used in the ontology.

    Args:
        term: The term to search for.

    Returns:
        A list of tuples, each containing a CL ID and a label.
    """
    adapter = get_adapter("ols:cl")
    # print("Agent query:", term)
    results = adapter.basic_search(term)
    results = list(adapter.labels(results))
    
    # remove any terms imported from other ontologies
    results = list(filter(lambda r: (r[0] is not None) and (r[0].startswith("CL:")), results))
    results = results[:20] # limit to first 20 results
    if len(results) == 0:
        results = "No results were found. Try another query."
        # without adding this text, the model sometimes repeatedly makes the same call
    print(f"## Agent query: {term} -> {results}")
    return results



def link_text(agent, text: str) -> CL_identifier:
    print("Term to link:", text)
    result = agent.run_sync(text)
    print("Linked to", result.output, end='\n\n')
    return result.output

async def async_link_text(agent, text: str) -> CL_identifier:
    result = await agent.run(text)
    return result.output

# print("fibroblast:", link_text("fibroblast"))
# print("endothelial progenitor cell:", link_text("endothelial progenitor cell"))
# print("L5 extratelencephalic projecting glutamatergic cortical neuron:", link_text("L5 extratelencephalic projecting glutamatergic cortical neuron"))

