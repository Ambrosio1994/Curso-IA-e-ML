# LLM com Langchain

Este diretório contém projetos práticos utilizando Large Language Models (LLMs) com o framework Langchain.

## Projetos

### Projeto 01 - Transcrição e Compreensão de Vídeos
- Implementação de sistema para transcrição automática de vídeos
- Análise e compreensão do conteúdo transcrito
- Geração de resumos e insights

### Projeto 02 - Chatbot Customizado
- Implementação de chatbot com memória de contexto
- Interface gráfica usando Streamlit
- Sistema de gerenciamento de histórico de conversas

### Projeto 03 - RAG Avançada
- Sistema de Retrieval Augmented Generation
- Capacidade de consulta a múltiplos documentos
- Compreensão contextual avançada
- Interface interativa para consultas

## Requisitos
- Python 3.8+
- Langchain
- OpenAI API key ou modelo local (como Ollama)
- Streamlit
- FAISS para indexação de documentos

## Instalação
```bash
pip install langchain openai streamlit faiss-cpu
```

## Como Usar
1. Configure suas chaves de API no arquivo .env
2. Execute os scripts específicos de cada projeto:
```bash
streamlit run curso_llms_projeto_02_chatbot_customizado_com_memória_e_interface.py
``` 