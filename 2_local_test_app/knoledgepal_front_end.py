import gradio as gr
from llm_inference import knowledgepal

def knowledge_pal_response(message, history):
    
    sys = """You are "Knowledge pal", an assistant for developers. Your role consists on answering questions about cloud services, coding in different languages, and provide a detailed response everytime. You will provide the source of the answer you are giving and related sources. Those sources can be a path or URL. Answer the question by combining your knowledge with the context provided. If information is not clear or the context is not enough to give an answer, tell the user that you don't have the answer. \n\n """

    prompt = chatbot.prompt_gen(sys, message, 2)
    answer = chatbot.rag_inference(prompt, False, 0.2, 0.6, 512, None)

    return answer

demo = gr.ChatInterface(knowledge_pal_response)

if __name__ == "__main__":

    models_folder = ".\\1_Tests\\hf_models"
    model_name = "TheBloke/Llama-2-7B-Chat-GGML"
    model_filename = "llama-2-7b-chat.ggmlv3.q4_K_M.bin"

    llama_model_path = f"{models_folder}\\{model_filename}"

    chatbot = knowledgepal(llama_model_path, 4096, 5)

    demo.launch()
