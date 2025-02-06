# Mineração de Emoção em Texto

Este diretório contém implementações de técnicas para análise e mineração de emoções em textos utilizando processamento de linguagem natural.

## Projetos

### Lematização
- Implementação de técnicas de lematização para português
- Uso do spaCy com modelo pt_core_news_sm
- Normalização de textos para análise

### Mineração de Texto
- Pré-processamento de texto
- Remoção de stopwords
- Tokenização
- Análise de frequência de palavras

### Sumarização de Texto
Diferentes abordagens para sumarização:
- Sumarização baseada em cosseno
- Sumarização usando pysummarization
- Sumarização usando Sumy
- Implementação de funções customizadas

## Tecnologias Utilizadas
- NLTK
- spaCy
- pysummarization
- sumy
- NumPy
- Pandas

## Como Executar
1. Instale as dependências:
```bash
pip install nltk spacy pysummarization sumy
python -m spacy download pt_core_news_sm
```

2. Execute os notebooks Jupyter correspondentes a cada técnica 