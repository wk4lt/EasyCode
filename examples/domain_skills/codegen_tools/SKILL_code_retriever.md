# Skill: code_retriever

## Description
Query the RAG vector database to find similar code examples based on a design document or query. Returns the most relevant design+code pairs with similarity scores.

## Parameters
- query_text (string): The search query (typically a design document or requirement description). Required.
- n_results (integer): Number of results to return, between 1 and 10. Optional, default=5.

## Boundaries
- Max query length: 50000 characters
- Results only returned if the RAG store has been populated with examples
