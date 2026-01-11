"""
To work, this needs:
    - cell_types.tsv (CL names and identifiers)
    - abbreviations file
"""


import sys
import json

import numpy as np
import torch
from tqdm.auto import tqdm            
from transformers import AutoTokenizer, AutoModel  
from scipy.spatial.distance import cdist
import bioc
import os
import openai
import time
from s_stem import s_stem_all
import abbreviations
import oak_agent
import asyncio

if len(sys.argv) != 6:
    raise Exception("Usage: python normalize.py <input_xml_filepath> <output_xml_filepath> <cell_types_tsv_filepath> " + \
                    "<abbreviations_path> <OpenAI_embeddings_cache_path>")

input_xml_filepath = sys.argv[1]
output_xml_filepath = sys.argv[2]
cell_types_tsv_filepath = sys.argv[3]
abbreviations_path = sys.argv[4]
OpenAI_embeddings_cache_path = sys.argv[5]

topn = 10

# model_names = {
#     "SapBERT": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
#     "MedCPT-Query": "ncbi/MedCPT-Query-Encoder",
#     "OpenAI-txt-emb-3-L": "text-embedding-3-large",
#     "GPT-5.2_Agent": "gpt-5.2",
#     }

model_names = {
    "GPT-5.2_Agent": "gpt-5.2",
    }


def load_terms(filename):
    term_id_pairs = list()
    with open(filename, "r") as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            fields = line.split("\t")
            if len(fields) != 2:
                raise ValueError()
            name = s_stem_all(fields[0])
            id = fields[1]
            term_id_pairs.append((name, id))
    return term_id_pairs

def encode_names(tokenizer, model, names, device):
    bs = 128
    all_reps = []
    with torch.no_grad():
        for i in tqdm(np.arange(0, len(names), bs)):
            toks = tokenizer.batch_encode_plus(names[i:i+bs], 
                                               padding="max_length", 
                                               max_length=25, 
                                               truncation=True,
                                               return_tensors="pt")
            toks = {k: v.to(device) for k, v in toks.items()}
            output = model(**toks)
            cls_rep = output[0][:,0,:]
            
            all_reps.append(cls_rep.cpu().detach().numpy())
    all_reps_emb = np.concatenate(all_reps, axis=0)
    return all_reps_emb

def topk(array, k, axis=-1, sorted=True):
    # Use np.argpartition is faster than np.argsort, but do not return the values in order
    # We use array.take because you can specify the axis
    partitioned_ind = (
        np.argpartition(array, -k, axis=axis)
        .take(indices=range(-k, 0), axis=axis)
    )
    # We use the newly selected indices to find the score of the top-k values
    partitioned_scores = np.take_along_axis(array, partitioned_ind, axis=axis)
    
    if sorted:
        # Since our top-k indices are not correctly ordered, we can sort them with argsort
        # only if sorted=True (otherwise we keep it in an arbitrary order)
        sorted_trunc_ind = np.flip(
            np.argsort(partitioned_scores, axis=axis), axis=axis
        )
        
        # We again use np.take_along_axis as we have an array of indices that we use to
        # decide which values to select
        ind = np.take_along_axis(partitioned_ind, sorted_trunc_ind, axis=axis)
        scores = np.take_along_axis(partitioned_scores, sorted_trunc_ind, axis=axis)
    else:
        ind = partitioned_ind
        scores = partitioned_scores
    
    return scores, ind
    

def run_query(tokenizer, model, all_reps_emb, term_id_pairs, query, device):
    query_toks = tokenizer.batch_encode_plus([query], padding="max_length", max_length=25, truncation=True, return_tensors="pt")
    query_toks = {k: v.to(device) for k, v in query_toks.items()}
    with torch.no_grad():
        query_output = model(**query_toks)
    query_cls_rep = query_output[0][:,0,:]
    dists = cdist(query_cls_rep.cpu().detach().numpy(), all_reps_emb, metric="cosine")
    return get_topn_data(dists, term_id_pairs)

def get_topn_data(dists, term_id_pairs):
    topn_scores, topn_indices = topk(-dists, topn)
    #print("topn_scores = {} topn_indices = {}".format(type(topn_scores), type(topn_indices)))
    #print("topn_scores = {} topn_indices = {}".format(topn_scores, topn_indices))
    topn_results = list()
    for ii in range(len(topn_indices[0])):
        #print("ii = {} type = {}".format(ii, type(ii)))
        index = topn_indices[0][ii]
        score = topn_scores[0][ii]
        #print("index = {} type = {}".format(index, type(index)))
        #print("score = {} type = {}".format(score, type(score)))
        name, identifier = term_id_pairs[index]
        #print("identifier = {} type = {}".format(identifier, type(identifier)))
        topn_results.append((name, identifier, score))
    return topn_results


def get_mention_texts(input_filename):
    mentions = set()
    print("Loading mention texts from file " + input_filename)
    with open(input_filename, "r") as fp:
        input_collection = bioc.load(fp)
    for document in input_collection.documents:
        for passage in document.passages:
            if not passage.infons.get('annotatable', True):
                continue
            pmid = passage.infons.get("article-id_pmid", None)
            # if pmid is None:
            #     print("warning/take a look: passage has no pmid", passage)
            #     continue
            for annotation in passage.annotations:
                if annotation.infons['type'] != 'cell_desc':
                    mentions.add((pmid, s_stem_all(annotation.text)))
    return mentions

def process_collection(input_filename, models_results, output_filename):
    print("Processing file " + input_filename + " to " + output_filename)
    with open(input_filename, "r") as fp:
        bioc_collection = bioc.load(fp)
    # output_collection = bioc.BioCCollection()
    for document in bioc_collection.documents:
        for passage in document.passages:
            if not passage.infons.get('annotatable', True) or (len(passage.text)==0):
                continue
            pmid = passage.infons.get("article-id_pmid")
            if pmid is None:
                print("warning/take a look: passage has no pmid", passage)
                continue
            for annotation in passage.annotations:
                if annotation.infons['type'] != 'cell_desc':
                    for model_name, normalized in models_results.items():
                        topn_results = normalized.get((pmid, s_stem_all(annotation.text)))
                        if topn_results is None:
                            print("WARN: No normalized identifier found for pmid = {} mention text = {}".format(pmid, s_stem_all(annotation.text)))
                        else:
                            for index in range(topn):
                                identifier = topn_results[index][1] if index < len(topn_results) else "-"
                                if index == 0:
                                    identifier_score = 1.0 + topn_results[index][2]
                                    identifier_score *= identifier_score
                                    annotation.infons[model_name + "_identifier_score_" + str(index)] = identifier_score
                                    annotation.infons[model_name + "_identifier_name_" + str(index)] = topn_results[index][0]
                                annotation.infons[model_name + "_id_" + str(index)] = identifier
        # output_collection.add_document(document)
    with open(output_filename, "w") as fp:
        # bioc.dump(output_collection, fp)
        bioc.dump(bioc_collection, fp)




def paths_to_filenames(input_paths, output_paths):
    new_input_paths = []
    new_output_paths = []
    for i, (input_path, output_path) in enumerate(zip(input_paths, output_paths)):
        if os.path.isdir(input_path) and os.path.isdir(output_path):
            # if input_path == output_path, then files will be overwritten
            new_input_paths  += [os.path.join(input_path, filename)  for filename in os.listdir(input_path)]
            new_output_paths += [os.path.join(output_path, filename) for filename in os.listdir(input_path)]
        elif os.path.isfile(input_path) and not os.path.isdir(output_path):
            new_input_paths.append(input_path)
            new_output_paths.append(output_path)
        else:
            raise Exception("both input and output path must be either directory or file")
    return new_input_paths, new_output_paths


def run_inference(model_name, term_id_pairs, mentions, abbr):
    # Load model
    print("Loading model:", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)  
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    
    model = model.to(device)
    model.eval()

    # Encode terms
    print("Encoding names")
    dictionary_names = [p[0] for p in term_id_pairs]
    all_reps_emb = encode_names(tokenizer, model, dictionary_names, device)
    
    
    normalized = dict()
    query_cache = dict()
    expanded_text_dict = dict()
    # start = datetime.datetime.now()
    for index, (document_id, mention_text) in enumerate(tqdm(mentions)):
        expanded_text = abbr.expand(document_id, mention_text, expanded_text_dict)
        if expanded_text in query_cache:
            topn_results = query_cache[expanded_text]
        else:
            topn_results = run_query(tokenizer, model, all_reps_emb, term_id_pairs, expanded_text, device)
            query_cache[expanded_text] = topn_results
        normalized[(document_id, mention_text)] = topn_results
    del model
    return normalized


def OpenAI_embed(client, model_name, text, attempts=5):
    if model_name not in OpenAI_embeddings_cache:
        OpenAI_embeddings_cache[model_name] = dict()
    if text in OpenAI_embeddings_cache[model_name]:
        return OpenAI_embeddings_cache[model_name][text]
    else:
        for attempt in range(attempts):
            try:
                print("querying:", text)
                emb = client.embeddings.create(input = [text], model=model_name).data[0].embedding
                OpenAI_embeddings_cache[model_name][text] = emb
                return emb
            except Exception as e:
                print("got exception", e)
                print("re-querying", text)
                time.sleep(2)
        raise Exception(f"Could not get embeddings after {attempts} attempts")

def OpenAI_batch_embed(client, model_name, texts, attempts=5, batch_size=100):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i : i + batch_size]
        embeddings += OpenAI_embed_many(client, model_name, batch, attempts)
        time.sleep(1)
    return embeddings
        

def OpenAI_embed_many(client, model_name, batch, attempts):
    for attempt in range(attempts):
        try:
            # print("querying:", text)
            response = client.embeddings.create(input = batch, model=model_name)
            embs = [item.embedding for item in response.data]
            return embs
        except Exception as e:
            print("got exception", e)
            print("re-querying", batch)
            time.sleep(2)
    raise Exception(f"Could not get embeddings after {attempts} attempts")


def run_OpenAI_embeddings_inference(model_name, term_id_pairs, mentions, abbr, embeddings_cache_path):
    # Load model
    print("Loading GPT client.")
    
    client = openai.AzureOpenAI(
      azure_endpoint = os.environ['endpoint_embeddings'],
      api_key = os.environ['api_key_embeddings'],
      api_version = "2025-02-01-preview"  
    )
    
    
    # Encode terms
    print("Encoding names")
    dictionary_terms = [p[0] for p in term_id_pairs]
    # print("embedding vocab")
    # all_reps_emb = np.asarray([OpenAI_embed(client, model_name, term) for term in dictionary_names])
    all_reps_emb = np.asarray(OpenAI_batch_embed(client, model_name, dictionary_terms))
    # print("finished emedding vocab")
    
    
    all_expanded_mentions = list(set([abbr.expand(doc_id, mention_text) for (doc_id, mention_text) in mentions]))
    all_mentions_emb = OpenAI_batch_embed(client, model_name, all_expanded_mentions)
    all_mentions_emb_dict = {expanded_mention: emb for (expanded_mention, emb) in zip(all_expanded_mentions, all_mentions_emb)}
    
    normalized = dict()
    query_cache = dict()
    expanded_text_dict = dict()
    # start = datetime.datetime.now()
    for index, (document_id, mention_text) in enumerate(tqdm(mentions, desc="Comparing OpenAI emeddings similarities")):
        # print("mention:", mention_text)
        expanded_text = abbr.expand(document_id, mention_text, expanded_text_dict)
        if expanded_text in query_cache:
            topn_results = query_cache[expanded_text]
        else:
            query_cls_rep = np.expand_dims(all_mentions_emb_dict[expanded_text], 0)
            dists = cdist(query_cls_rep, all_reps_emb, metric="cosine")
            topn_results = get_topn_data(dists, term_id_pairs)
            query_cache[expanded_text] = topn_results
            # if (index % 5 == 0) or (index == len(mentions) - 1):
            #     # save cache
            #     with open(OpenAI_embeddings_cache_path, 'w') as writefp:
            #         json.dump(OpenAI_embeddings_cache, writefp)
        normalized[(document_id, mention_text)] = topn_results
    return normalized
    
    # normalized = dict()
    # query_cache = dict()
    # expanded_text_dict = dict()
    # # start = datetime.datetime.now()
    # for index, (document_id, mention_text) in enumerate(tqdm(mentions)):
    #     # print("mention:", mention_text)
    #     expanded_text = abbr.expand(document_id, mention_text, expanded_text_dict)
    #     if expanded_text in query_cache:
    #         topn_results = query_cache[expanded_text]
    #     else:
    #         query_cls_rep = np.expand_dims(OpenAI_embed(client, model_name, expanded_text), 0)
    #         dists = cdist(query_cls_rep, all_reps_emb, metric="cosine")
    #         topn_results = get_topn_data(dists, term_id_pairs)
    #         query_cache[expanded_text] = topn_results
    #         if (index % 5 == 0) or (index == len(mentions) - 1):
    #             # save cache
    #             with open(OpenAI_embeddings_cache_path, 'w') as writefp:
    #                 json.dump(OpenAI_embeddings_cache, writefp)
    #     normalized[(document_id, mention_text)] = topn_results
    # return normalized

async def agent_batch_process(agent, texts: list[str]) -> list[str]:
    tasks = [oak_agent.async_link_text(agent, t) for t in texts]
    results = []
    
    import tqdm.asyncio
    for coro in tqdm.asyncio.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="EL agents"):
        res = await coro
        results.append(res)
    return results

# def run_agent_inference(model_name, term_id_pairs, mentions, abbr):
#     # set up agent
#     agent = oak_agent.set_up_model(model_name)
    
#     cache = OpenAI_embeddings_cache.get(model_name + "_Agent", dict())
    
#     all_expanded_mentions = list(set([abbr.expand(doc_id, mention_text) for (doc_id, mention_text) in mentions]))
#     mentions_to_link = [mention for mention in all_expanded_mentions if mention not in cache]
#     CL_identifiers = asyncio.run(agent_batch_process(agent, mentions_to_link))
    
#     cache.update({mention: ID_object.identifier for (mention, ID_object) in zip(mentions_to_link, CL_identifiers)})
    
#     # save cache
#     OpenAI_embeddings_cache[model_name + "_Agent"] = cache
#     with open(OpenAI_embeddings_cache_path, 'w') as writefp:
#         json.dump(OpenAI_embeddings_cache, writefp)
    
#     all_mentions_linked_dict = cache
    
    
#     normalized = dict()
#     expanded_text_dict = dict()
#     for index, (document_id, mention_text) in enumerate(tqdm(mentions, desc="Saving agent results")):
#         expanded_text = abbr.expand(document_id, mention_text, expanded_text_dict)
#         cached_result = all_mentions_linked_dict[expanded_text]
#         if type(cached_result) == list:
#             top_ID = cached_result[1]
#             topn_results = cached_result
#         else:
#             top_ID = cached_result
#             topn_results = [("-", top_ID, 0)]
#         normalized[(document_id, mention_text)] = topn_results
#     return normalized

def run_agent_inference(model_name, term_id_pairs, mentions, abbr):
    # set up agent
    agent = oak_agent.set_up_model(model_name)
        
    normalized = dict()
    query_cache = OpenAI_embeddings_cache.get(model_name + "_Agent", dict())
    expanded_text_dict = dict()
    # start = datetime.datetime.now()
    new_terms_counter = 0
    for index, (document_id, mention_text) in enumerate(tqdm(mentions)):
        expanded_text = abbr.expand(document_id, mention_text, expanded_text_dict)
        if expanded_text in query_cache:
            # if mention_text != expanded_text:
            #     print(f"using cache ({mention_text} / {expanded_text})")
            # else:
            #     print(f"using cache ({mention_text})")
            topn_results = query_cache[expanded_text]
        else:
            # if mention_text != expanded_text:
            #     print(f"querying: {mention_text} via {expanded_text}")
            # else:
            #     print("querying:", expanded_text)
            for attempt in range(5):
                try:
                    top_ID = oak_agent.link_text(agent, expanded_text).identifier
                    break
                except Exception as e:
                    print("attempt", attempt+1, "failed for mention", expanded_text)
                    print("exception", e)
                    if attempt == 4:
                        raise
            topn_results = [("-", top_ID, 0)]
            query_cache[expanded_text] = topn_results
            new_terms_counter += 1
            if (new_terms_counter == 20) or (index == len(mentions) - 1):
                # save cache
                OpenAI_embeddings_cache[model_name + "_Agent"] = query_cache
                with open(OpenAI_embeddings_cache_path, 'w') as writefp:
                    json.dump(OpenAI_embeddings_cache, writefp)
                new_terms_counter = 0
        normalized[(document_id, mention_text)] = topn_results
    return normalized


def main(term_filename, abbr_freq_filename, abbr_paths, input_paths, output_paths, model_names, OpenAI_embeddings_cache_path):
    if type(abbr_paths) == str:
        abbr_paths = [abbr_paths]
    if type(input_paths) == str:
        input_paths = [input_paths]
    if type(output_paths) == str:
        output_paths = [output_paths]
    
    # if given a directory, then make it a file
    input_paths, output_paths = paths_to_filenames(input_paths, output_paths)
    # could delete the multi-file functionality above if desired
    
    
    # Load abbreviations
    if (abbr_freq_filename == "" or abbr_freq_filename == "."):
        print("No abbreviation frequencies")
        abbr_freq_dict = dict()
    else:
        with open(abbr_freq_filename, "r") as abbr_freq_file:
            abbr_freq_dict = json.load(abbr_freq_file) # Load the abbreviation frequency file
    
    print("Loading abbreviations")
    abbr = abbreviations.AbbreviationExpander(abbr_freq_dict)
    for abbr_path in abbr_paths:
        abbr.load(abbr_path)

    print("Loading terms")
    term_id_pairs = load_terms(term_filename)
    # print(term_id_pairs)
    print("Loaded {} terms from the reference vocabulary.".format(len(term_id_pairs)))
    
    
    # Load mentions
    mentions = set()
    for input_filename, output_filename in zip(input_paths, output_paths):
        mentions.update(get_mention_texts(input_filename))
    
    
    models_results = dict()
    for model_nickname, model_fullname in model_names.items():
        if "agent" in model_nickname.lower():
            models_results[model_nickname] = run_agent_inference(model_fullname, term_id_pairs, mentions, abbr)
        elif "openai" in model_nickname.lower():
            # print("going to run run_OpenAI_embeddings_inference()")
            models_results[model_nickname] = run_OpenAI_embeddings_inference(model_fullname, term_id_pairs, mentions, abbr, OpenAI_embeddings_cache_path)
            # print("finished run_OpenAI_embeddings_inference()")
        else:
            models_results[model_nickname] = run_inference(model_fullname, term_id_pairs, mentions, abbr)
    
    print("going to process files")
    for input_filename, output_filename in zip(input_paths, output_paths):
        process_collection(input_filename, models_results, output_filename)
    
    print("Done.")
    return


if os.path.isfile(OpenAI_embeddings_cache_path):
    print("Loading cached GPT embeddings.")
    with open(OpenAI_embeddings_cache_path) as readfp:
        OpenAI_embeddings_cache = json.load(readfp)
else:
    print("No cached embeddings.")
    OpenAI_embeddings_cache = dict()


main(cell_types_tsv_filepath, '', abbreviations_path, input_xml_filepath,
      output_xml_filepath, model_names, OpenAI_embeddings_cache_path)


with open(OpenAI_embeddings_cache_path, 'w') as writefp:
    json.dump(OpenAI_embeddings_cache, writefp)
print("saved OpenAI embeddings")
