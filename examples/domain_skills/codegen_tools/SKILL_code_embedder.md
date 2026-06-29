# Skill: code_embedder

## Description
Embed a code example (design document + implementation code pair) and store it in the RAG vector database for later retrieval. This is used during the learning phase to index examples. Prefer passing file paths (design_path, code_path) over content strings to save tokens.

## Parameters
- design_path (string): Path to the design document (.md) to read and embed. Preferred over design_content.
- code_path (string): Path to the implementation file (.py) to read and embed. Preferred over code_content.
- design_content (string): The design document content (markdown format). Only used if design_path is empty.
- code_content (string): The implementation code content. Only used if code_path is empty.
- metadata_json (string): Optional JSON string with extra metadata (source paths, labels, etc.). Optional, default={}.
- doc_id (string): Optional explicit document ID. Auto-generated if not provided. Optional.

## Boundaries
- Max design_content length: 50000 characters
- Max code_content length: 100000 characters
- Maximum of 10000 documents in the collection
