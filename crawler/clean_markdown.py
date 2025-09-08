import re


def process_markdown_results(results):
    """
    Process all crawled results with various cleaning operations.

    This function acts as an orchestrator for different markdown cleaning
    operations. It can remove links and perform other cleaning tasks.

    Args:
        results (list): List of crawler result objects

    Returns:
        list: Processed results with all cleaning operations applied
    """
    # Remove all markdown links
    processed_results = remove_all_markdown_links(results)

    # Future cleaning operations can be added here
    # processed_results = another_cleaning_function(processed_results)

    return processed_results


def remove_all_markdown_links(results):
    """
    Remove all markdown links and images from all documents, preserving only the link
    text for regular links and removing images entirely.

    Args:
        results (list): List of crawler result objects

    Returns:
        list: Processed results with all links and images removed
    """
    processed_results = []

    # Regular expression to identify markdown links and images
    # This matches both [text](url) and ![alt](url) formats
    link_pattern = re.compile(r"!?\[(.*?)\]\((.*?)\)")

    for result in results:
        # Determine which markdown content to use
        markdown_content = None

        # Check for result.markdown first
        if hasattr(result, "markdown") and isinstance(result.markdown, str):
            markdown_content = result.markdown
        # Check for result.markdown.fit_markdown as fallback
        elif (
            hasattr(result, "markdown")
            and hasattr(result.markdown, "fit_markdown")
            and isinstance(result.markdown.fit_markdown, str)
        ):
            markdown_content = result.markdown.fit_markdown

        # Skip if no valid markdown content found
        if not markdown_content:
            processed_results.append(result)
            continue

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
        
        modified_content = link_pattern.sub(replace_link_or_image, markdown_content)

        # Update the markdown content (always set result.markdown)
        result.markdown = modified_content
        processed_results.append(result)

    return processed_results
