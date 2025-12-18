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
        You are an expert ontology curator. Use the ontologies at your disposal to
        answer the user's questions.
        """,
        tools=[search_CL],
        result_type=CL_identifier,
        model_settings={"do_sample": False}
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
    print("Agent query:", term)
    results = adapter.basic_search(term)
    labels = list(adapter.labels(results))
    # print(f"## Query: {term} -> {labels}")
    return labels



def link_text(agent, text: str) -> CL_identifier:
    result = agent.run_sync(text)
    return result.output

async def async_link_text(agent, text: str) -> CL_identifier:
    result = await agent.run(text)
    return result.output

# print("fibroblast:", link_text("fibroblast"))
# print("endothelial progenitor cell:", link_text("endothelial progenitor cell"))
# print("L5 extratelencephalic projecting glutamatergic cortical neuron:", link_text("L5 extratelencephalic projecting glutamatergic cortical neuron"))

