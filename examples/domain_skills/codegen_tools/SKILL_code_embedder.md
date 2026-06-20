# Skill: code_embedder

## Description
Embed a code example (design document + implementation code pair) and store it in the RAG vector database for later retrieval. This is used during the learning phase to index examples.

## Parameters
- design_content (string): The design document content (markdown format). Required.
- code_content (string): The implementation code content. Required.
- metadata_json (string): Optional JSON string with extra metadata (source paths, labels, etc.). Optional, default={}.
- doc_id (string): Optional explicit document ID. Auto-generated if not provided. Optional.

## Boundaries
- Max design_content length: 50000 characters
- Max code_content length: 100000 characters
- Maximum of 10000 documents in the collection
