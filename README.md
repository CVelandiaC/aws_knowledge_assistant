# Knowledge Pal - AWS Assistant

__Author:__ Cristian C. Velandia C.

---

__Goal:__ Develop an AI based assistant to help developers with their questions about documentation. The assistant must be mainly focused on AWS services nad internal processes. 

## __Solution:__

Two solutions based on retrieval augmented generation (RAG) architure are tested, one with local llama quantized cpp model and other with OpanAi API. Both are tested in notebooks, but just the llama one is selected for local testing of the assistant.

## Folders description

__1_Tests:__ This folder stores the local test notebook in which the models, hyperparameters and strategies are tried and tested. Two strategies can be found here, notebooks based on OpenAI model and other notebooks with the final solution based on llama.

__2_local_test_app:__ This folder stores the local test script which uses gradio to interact with the RAG system. To run it you must download the model from hugging face, the script can be found at folder 1 in the "2_2_test_embeddings_llama.ipynb" notebook at the "Setup Llama Embeddings" section.  

### References
* https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML?sagemaker_deploy=true
* https://python.langchain.com/docs/integrations/text_embedding/sagemaker-endpoint
* https://docs.pinecone.io/docs/indexes
* https://docs.ragas.io/en/stable/getstarted/monitoring.html#get-started-monitoring
* https://python.langchain.com/docs/get_started/introduction
* https://docs.aws.amazon.com/
* https://huggingface.co/docs
* https://www.gradio.app/guides/creating-a-chatbot-fast

