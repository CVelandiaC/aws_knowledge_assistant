from pinecone import Pinecone, ServerlessSpec
from llama_cpp import Llama
from tqdm.auto import tqdm
import pandas as pd
import json

from langchain.embeddings import LlamaCppEmbeddings
from llama_cpp import Llama

class knowledgepal:

    def __init__(self, model_path, nctx, n_gpu_layers):
        #get credentials
        creds =json.load(open('.\\1_Tests\\personal_creds.json'))

        #Setup pinecone
        PINECONE_API_KEY = creds["PINECONE_API_KEY"]
        INDEX_NAME = 'aws-docs-vdb-index'

        pinecone = Pinecone(api_key = PINECONE_API_KEY)

        self.index = pinecone.Index(INDEX_NAME)

        #Instantiate models
        self.llm_model = Llama(model_path, n_ctx = nctx)
        self.emb_model = LlamaCppEmbeddings(model_path = model_path, n_ctx = nctx, n_gpu_layers = n_gpu_layers)


    def rag_inference(self, message, stream, temp, top, max, stop):

        knowledge_pal_response_gen = self.llm_model(prompt = message, stream = stream, temperature = temp, top_p = top, max_tokens = max, stop = stop)
        if stream:
            knowledge_pal_response = "".join([c["choices"][0]["text"] for c in knowledge_pal_response_gen])
        else:
            knowledge_pal_response = knowledge_pal_response_gen["choices"][0]["text"]

        return knowledge_pal_response
    
    def prompt_gen(self, sys, query, top):
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        SYSTEM_BEHAVIOR = sys

        embs = self.emb_model.embed_query(query)
        res = self.index.query(vector = embs, top_k = top, include_metadata = True)
        context = [r['metadata']['text'] for r in res['matches']]

        message = B_INST + B_SYS + SYSTEM_BEHAVIOR + "Context: " + '\n'.join(context) + E_SYS + "Question: " + query + " Answer: " + E_INST

        return message