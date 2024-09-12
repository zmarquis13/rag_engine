### RAG Engine

A program that easily and efficiently turns any pdf (or set of pdfs) into a chattable RAG and updates automatically with new PDFs. When questions cannot be answered from the contextual data, the program performs a web search to provide more information. The user is let known whenever a web search occurs.

Potential use cases:
* chatting over textbook pdfs for courses
* AI tutoring in a specific topic or set of topics
* basing answers on a small set of trusted sources to maximize reliability
* finding specific quotes in a large body of text

Usage:
* clone the repo
* set OPENAI_API_KEY and BRAVE_SEARCH_API_KEY in a .env file in the root directory
* put any pdfs you want to chat over in a folder called new_pdfs
* type the command ```jupyter lab``` and run the only cell in rag_engine.ipynb (```python rag_engine.py``` to come soon) 
* chat away!

Next steps:
* incorporate support for more file types (txt, etc.)
* allow the content of specific webpages to be indexed as well




