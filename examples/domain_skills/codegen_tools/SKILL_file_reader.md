# Skill: file_reader

## Description
Read the contents of a file from the local filesystem. Returns the raw text content. Supports reading design documents (.md), implementation code (.py), and test files.

## Parameters
- file_path (string): The absolute or relative path to the file to read. Required.

## Boundaries
- Max file size: 1MB
- Only reads text files (.md, .py, .txt, .yaml, .json, .toml)
- Encoding: UTF-8
