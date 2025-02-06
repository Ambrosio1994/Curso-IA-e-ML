
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import HuggingFaceHub

from dotenv import load_dotenv

load_dotenv()

#  Configurações do Streamlit
st.set_page_config(page_title="Seu assistente virtual 🤖", page_icon="🤖")
st.title("Seu assistente virtual 🤖")

model_class = "hf_hub" 

def model_hf_hub(model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.1):
  llm = HuggingFaceHub(
      repo_id=model,
      model_kwargs={
          "temperature": temperature,
          "return_full_text": False,
          "max_new_tokens": 100,
          #"stop": ["<|eot_id|>"],
          # demais parâmetros que desejar
      }
  )
  return llm


def model_openai(model="gpt-4o-mini", temperature=0.1):
    llm = ChatOpenAI(
        model=model,
        temperature=temperature
    )
    return llm

def model_ollama(model="phi3", temperature=0.1):
    llm = ChatOllama(
        model=model,
        temperature=temperature,
    )
    return llm
# 
# 
def model_response(user_query, chat_history, model_class):

    ## Carregamento da LLM
    if model_class == "hf_hub":
        llm = model_hf_hub()
    elif model_class == "openai":
        llm = model_openai()
    elif model_class == "ollama":
        llm = model_ollama()

    ## Definição dos prompts
    system_prompt = """
    Você é um assistente prestativo e está respondendo perguntas gerais. Responda em {language}.
    """
    # corresponde à variável do idioma em nosso template
    language = "português"

    # Adequando à pipeline
    if model_class.startswith("hf"):
        user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n{input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    else:
        user_prompt = "{input}"

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", user_prompt)
    ])

    ## Criação da Chain
    chain = prompt_template | llm | StrOutputParser()

    ## Retorno da resposta / Stream
    return chain.stream({
        "chat_history": chat_history,
        "input": user_query,
        "language": language
    })

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(
            content="Olá, sou o seu assistente virtual! Como posso ajudar você?"
            )
    ]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

user_query = st.chat_input("Digite sua mensagem aqui...")

if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))

    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        resp = st.write_stream(model_response(user_query, st.session_state.chat_history, model_class))
        print(st.session_state.chat_history)

    st.session_state.chat_history.append(AIMessage(content=resp))