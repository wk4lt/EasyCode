# Skill: web_search

## Description
Search the web for information using DuckDuckGo Instant Answer API. Returns relevant results including abstracts, topics, and related links.

## Parameters
- query (string): The search query string to look up on the web. Required.
- max_results (integer): Maximum number of results to return, between 1 and 10. Optional, default=5.

## Boundaries
- Max query length: 500 characters
- Rate limit: 50 calls per minute per agent
- Results limited to abstract, topics, and related URLs
