#!/usr/bin/env python3
"""
Chunk Keyword Generation Module

This module processes chunked documents directly in memory,
evaluates their content, and adds relevant keywords to useful content.
It doesn't perform any file I/O operations.
"""

import concurrent.futures
from typing import List, Tuple
from openai import OpenAI
from langchain_core.documents import Document
from crawler.clean_markdown import remove_all_markdown_links

client = OpenAI()

# System prompt for content evaluation with structured output
system_prompt = """
You are an expert at evaluating content and generating relevant keywords.
Your task is to ONLY delete content that is "completely useless".
ALWAYS KEEP content unless it is:
- Completely empty
- Content that is not useful for the user
- Only social media buttons/links
- Only generic 'Contact Us' or 'Menu' text

For content worth keeping, generate 5-10 KEYWORDS that:
- DO NOT already exist in the content
- Are semantically related to the content
- Would help a vector search engine find this content
- Use precise terminology for RAG retrieval
"""


def summarize_content(chunked_documents: List[Document], config=None):
    """Process chunked documents concurrently and return results with keywords.

    Args:
        chunked_documents (list): List of Document objects with chunk content
        config: Optional configuration object with summarization parameters

    Returns:
        list: Filtered documents with added keywords
    """
    print(f"Found {len(chunked_documents)} chunks to process")

    if not chunked_documents:
        print("No chunks found!")
        return []

    if config:
        model_name = getattr(config, "summary_model_name")
        temperature = getattr(config, "summary_temperature")
        max_workers = getattr(config, "summary_max_workers")

    processed_results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_chunk = {
            executor.submit(process_chunk, chunk, model_name, temperature): chunk
            for chunk in chunked_documents
        }
        for future in concurrent.futures.as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            status, processed_chunk = future.result()
            url = chunk.metadata.get("url", "unknown")
            print(f"Processed chunk from {url} - {status}")
            if processed_chunk is not None:
                processed_results.append(processed_chunk)

    return processed_results


def process_chunk(chunk, model_name, temperature):
    """Process an individual chunk.

    Args:
        chunk: A Document object with page_content and metadata
        model_name: The OpenAI model to use
        temperature: The temperature to use for the model

    Returns:
        tuple: (status, processed_chunk) where status is one of
        "kept", "deleted", "skipped", or "error"
    """
    try:
        # Get content from the chunk
        content = chunk.page_content
        url = chunk.metadata.get("url", "unknown")

        # Clean the content to remove links and images
        cleaned_content = clean_chunk_content(content)

        # Process content using the API
        keep, keywords = process_chunk_content(cleaned_content, client, model_name, temperature)

        if not keep:
            print(f"Marked for deletion: chunk from {url} - no useful content")
            return "deleted", None
        else:
            # Create a new result object similar to the crawl results
            # to maintain compatibility with downstream processing
            from types import SimpleNamespace

            result = SimpleNamespace()

            # Add the metadata from the original chunk
            for key, value in chunk.metadata.items():
                setattr(result, key, value)

            # Set standard attributes expected by downstream processes
            result.url = url
            result.markdown = (
                f"[View Source]({url})\n\n" f"Keywords: {keywords}\n\n" f"{cleaned_content}"
            )

            print(f"Updated chunk from {url} with keywords")
            return "kept", result
    except Exception as e:
        print(f"Error processing chunk from {url}: {e}")
        return "error", None


def process_chunk_content(
    content,
    client,
    model_name,
    temperature,
) -> Tuple[bool, str]:
    """Process chunk content and determine if it should be kept.

    Args:
        content (str): The content to process
        client: OpenAI client instance
        model_name: Model to use for processing
        temperature: The temperature to use for the model

    Returns:
        tuple: (keep_boolean, keywords) where keep_boolean is True if content
        should be kept or False if it should be deleted
    """
    try:
        # Verify content is not empty
        if not content or content.isspace():
            return False, ""

        # User prompt explaining the task
        user_prompt = (
            "Process this content and determine if it's worth keeping. "
            "If yes, provide relevant keywords. If not, return false "
            "for keep and empty string for keywords."
            f"\n\n{content}"
        )

        response = client.responses.create(
            input=user_prompt,
            instructions=system_prompt,
            model=model_name,
            temperature=temperature,
            service_tier="flex",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "content_evaluation",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "keep": {"type": "boolean"},
                            "keywords": {"type": "string"},
                        },
                        "required": ["keep", "keywords"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                }
            },
        )

        import json

        parsed_result = json.loads(response.output_text)

        print(f"Parsed result: {parsed_result}")

        keep = parsed_result.get("keep", False)
        keywords = parsed_result.get("keywords", "")

        return keep, keywords

    except Exception as e:
        print(f"Error processing chunk: {e}")
        return False, ""

def clean_chunk_content(content):
    """
    Clean chunk content by removing markdown links and images.
    
    Args:
        content (str): The chunk content to clean
        
    Returns:
        str: Cleaned content with links and images removed
    """
    import re
    
    # Regular expression to identify markdown links and images
    # This matches both [text](url) and ![alt](url) formats
    link_pattern = re.compile(r"!?\[(.*?)\]\((.*?)\)")
    
    # Replace all links and images
    # For regular links [text](url), keep the text content
    # For images ![alt](url), remove them entirely
    def replace_link_or_image(match):
        full_match = match.group(0)
        text_content = match.group(1)
        
        # If it's an image link (starts with !), remove it entirely
        if full_match.startswith('!'):
            return ""
        # If it's a regular link, keep the text content
        else:
            return text_content
    
    cleaned_content = link_pattern.sub(replace_link_or_image, content)
    return cleaned_content

