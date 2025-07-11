import asyncio
import os
import sys
from urllib.parse import urlparse, urlunparse
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMContentFilter,
    LLMConfig,
    DefaultMarkdownGenerator,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawler.sanitize_filename import sanitize_filename
from crawler.clean_markdown import process_markdown_results
from crawler.config import CrawlerConfig
from crawler.custom_markdown import create_custom_markdown_generator


async def crawl(config: CrawlerConfig = None):
    """
    Crawl websites based on provided configuration, process content,
    and return processed results.

    Args:
        config (CrawlerConfig, optional): Configuration object. If None,
                                         load from environment.

    Returns:
        list: Processed content results with duplicates removed.
    """
    # Safety check for max_depth
    if config.max_depth > 3:
        print(
            f"Warning: max_depth of {config.max_depth} is quite high and may cause recursion issues."
        )
        print("Consider reducing max_depth to 2-3 for better stability.")
        sys.setrecursionlimit(10000)

    deep_crawl = BFSDeepCrawlStrategy(
        max_depth=config.max_depth, include_external=config.include_external
    )

    crawler_link_config = CrawlerRunConfig(
        deep_crawl_strategy=deep_crawl,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        verbose=config.verbose,
        delay_before_return_html=config.delay_before_return_html,
        scan_full_page=True,
        # Use a simpler markdown generator for link extraction
        markdown_generator=None,  # Use default simple markdown
    )

    openai_config = LLMConfig(
        provider=config.llm_provider,
        api_token=os.environ.get("OPENAI_API_KEY"),
    )

    content_filter = LLMContentFilter(
        llm_config=openai_config,
        instruction=config.llm_instruction,
        verbose=config.verbose,
    )

    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    crawler_config = CrawlerRunConfig(
        markdown_generator=md_generator,
        excluded_tags=config.excluded_tags,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        verbose=config.verbose,
        delay_before_return_html=config.delay_before_return_html,
        scan_full_page=True,
        js_code=[
            config.exclude_hidden_elements and get_hidden_elements_removal_js() or None,
            get_dialogue_foundry_removal_js(),
            get_universal_structure_fix_js(),
        ],
    )

    # Crawler config for repeated elements to crawl once
    crawler_config_repeated_elements = CrawlerRunConfig(
        target_elements=config.excluded_tags,
        markdown_generator=md_generator,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        verbose=config.verbose,
        delay_before_return_html=config.delay_before_return_html,
        scan_full_page=True,
        js_code=[
            config.exclude_hidden_elements and get_hidden_elements_removal_js() or None,
            get_dialogue_foundry_removal_js(),
            get_universal_structure_fix_js(),
        ],
    )

    unique_links = set()
    all_results = []

    browser_config = BrowserConfig(
        browser_type=config.browser_type,
        headless=config.headless,
        light_mode=config.light_mode,
        text_mode=config.text_mode,
        ignore_https_errors=config.ignore_https_errors,
    )

    start_urls = config.start_urls

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # If no specific start URLs are provided, use a default
        if not config.start_urls:
            raise ValueError("No start URLs provided in configuration")
        else:
            start_urls = config.start_urls

        # First, crawl to collect all links
        for start_url in start_urls:
            print(f"Starting crawl from: {start_url}")
            response = await crawler.arun(
                start_url,
                config=crawler_link_config,
            )
            for results in response:
                for r in results:
                    # Debug: Print all link types found
                    print(f"Debug: All links found: {r.links}")

                    internal_links = r.links.get("internal", [])
                    print(f"Found {len(internal_links)} internal links")

                    # If no links found via markdown, try extracting from HTML directly
                    if len(internal_links) == 0 and len(r.html) > 0:
                        print(
                            "Debug: No links found in markdown, extracting from HTML..."
                        )
                        html_links = extract_links_from_html(r.html, start_url)
                        print(f"Debug: Found {len(html_links)} links in HTML")

                        # Convert HTML links to the expected format
                        internal_links = [{"href": link} for link in html_links]
                        print(
                            f"Debug: Converted to {len(internal_links)} internal links"
                        )

                    # Debug: Print first few internal links if any
                    if internal_links:
                        print(f"Sample internal links: {internal_links[:5]}")

                    for link in internal_links:
                        normalized_link = normalize_url(link["href"])
                        print(
                            f"Debug: Processing link {link['href']} -> "
                            f"{normalized_link}"
                        )

                        if not is_file_url(normalized_link):
                            print(f"Debug: Not a file URL: {normalized_link}")
                            if is_valid_web_url(normalized_link):
                                if should_exclude_path(
                                    normalized_link, config.excluded_paths
                                ):
                                    print(f"Debug: Excluding path: {normalized_link}")
                                else:
                                    print(
                                        f"Debug: Valid web URL, adding: "
                                        f"{normalized_link}"
                                    )
                                    unique_links.add(normalized_link)
                            else:
                                print(
                                    f"Debug: Invalid web URL, skipping: "
                                    f"{normalized_link}"
                                )
                        else:
                            print(f"Debug: File URL, skipping: {normalized_link}")

        print(f"Found {len(unique_links)} unique links.")

        # Define a helper function to crawl a single URL
        async def crawl_url(url):
            try:
                results = await crawler.arun(url, config=crawler_config)
                print(f"Completed: {url}")
                return results
            except Exception as e:
                print(f"Error crawling {url}: {e}")
                return []

        # Process URLs in batches
        batch_size = config.batch_size
        url_list = list(unique_links)
        total_batches = (len(url_list) + batch_size - 1) // batch_size

        print(f"Processing URLs in {total_batches} batches of {batch_size}")

        for batch_num in range(total_batches):
            start_idx = batch_num * batch_size
            end_idx = min(start_idx + batch_size, len(url_list))
            batch_urls = url_list[start_idx:end_idx]

            print(
                f"Processing batch {batch_num + 1}/{total_batches} "
                f"with {len(batch_urls)} URLs"
            )

            # Use asyncio.gather to process batch of URLs in parallel
            tasks = [crawl_url(url) for url in batch_urls]
            results_list = await asyncio.gather(*tasks)

            all_results.extend(results_list)

            print(f"Completed batch {batch_num + 1}/{total_batches}")

        for start_url in start_urls:
            results = await crawler.arun(
                start_url, config=crawler_config_repeated_elements
            )
            results.url = "repeated_elements"
            all_results.append(results)

    print(f"Crawling complete. Retrieved {len(all_results)} results.")

    # Debug: Check status codes of all results
    print("Debug: Status codes of all results:")
    for i, res in enumerate(all_results):
        status = getattr(res, "status_code", "NO_STATUS_CODE")
        url = getattr(res, "url", "NO_URL")
        print(f"  Result {i}: URL={url}, Status={status}")

    # Accept all 2xx success codes (200, 201, 202, 204, 206, etc.)
    valid_pages = [
        res
        for res in all_results
        if hasattr(res, "status_code")
        and res.status_code is not None
        and 200 <= res.status_code < 300
    ]
    print(f"Debug: After status filtering: {len(valid_pages)} valid pages")

    print("Post-processing results to remove links...")
    processed_pages = process_markdown_results(valid_pages)
    print(
        f"Debug: After process_markdown_results: "
        f"{len(processed_pages)} processed pages"
    )

    # Debug: Check if processed pages have markdown content
    print("Debug: Checking markdown content of processed pages:")
    for i, res in enumerate(processed_pages):
        url = getattr(res, "url", "NO_URL")
        markdown = getattr(res, "markdown", None)
        markdown_len = len(markdown) if markdown else 0
        print(f"  Processed {i}: URL={url}, Markdown len={markdown_len}")

    # Filter out empty content
    final_results = []
    custom_generator = create_custom_markdown_generator()

    for res in processed_pages:
        try:
            # Check if markdown generation failed (empty or just whitespace)
            if (
                not res.markdown
                or res.markdown.isspace()
                or len(res.markdown.strip()) <= 1
            ):
                print(
                    f"Markdown generation failed for {res.url}, trying custom generator..."
                )

                # Try custom markdown generator as fallback
                if hasattr(res, "html") and res.html:
                    custom_markdown = custom_generator.generate_markdown(
                        res.html, res.url
                    )
                    if custom_markdown and len(custom_markdown.strip()) > 10:
                        res.markdown = custom_markdown
                        print(f"✅ Custom markdown generation succeeded for {res.url}")
                    else:
                        print(
                            f"❌ Custom markdown generation also failed for {res.url}"
                        )
                        continue
                else:
                    print(f"❌ No HTML available for custom processing: {res.url}")
                    continue

            # Add metadata to the result object
            res.page_path = sanitize_filename(res.url)

            final_results.append(res)

        except Exception as e:
            print(f"❌ Failed to process page {res.url} due to: {e}")
            continue

    print(f"✅ Crawling complete! Processed {len(final_results)} unique pages")

    return final_results  # Return the processed results


def get_hidden_elements_removal_js():
    """Return JavaScript code that removes hidden elements from the DOM.

    This script runs in the browser context and removes elements that are
    hidden via CSS or HTML attributes to clean up the page before scraping.
    """
    return """
    (async () => {
        function isElementHidden(el) {
            const style = window.getComputedStyle(el);
            if (style.display === 'none' || 
                style.visibility === 'hidden') {
                return true;
            }
            if (el.getAttribute('hidden') !== null || 
                el.getAttribute('aria-hidden') === 'true') {
                return true;
            }
            return false;
        }

        if (document.body) {
            const elements = document.body.querySelectorAll('*');
            for (let el of elements) {
                if (isElementHidden(el)) {
                    el.remove();
                }
            }
        }
    })();
    """


def get_universal_structure_fix_js():
    """Return JavaScript that detects and fixes inverted content structures across various site builders.

    This script runs in the browser context to identify and reorder elements where content appears before its logical heading - a common issue in sites built with visual builders
    like Wix, Squarespace, etc.
    """
    return """
    (async () => {
        const baseTextSize = document.documentElement.style.fontSize;
        const baseTextSizePx = parseFloat(baseTextSize);
        
        function isLikelyHeading(el) {
            if (!el) return false;
            const text = el.textContent.trim();
            if (!text || text.length > 120) return false;

            if (el.querySelector('h1,h2,h3,h4,h5,h6')) return true;

            const style = window.getComputedStyle(el);
            const fontSize = parseFloat(style.fontSize);
            const fontWeight = parseInt(style.fontWeight);

            return (fontSize >= baseTextSizePx * 1.2 || fontWeight >= 600);
        }

        function isLikelyContent(el) {
            if (!el) return false;
            const text = el.textContent.trim();
            if (text.length > 150) return true;
            if (el.querySelectorAll('p, ul, ol, li').length >= 3) return true;

            return false;
        }

        // Go through all containers
        document.querySelectorAll('div, section, article').forEach(container => {
            const children = Array.from(container.children);
            for (let i = 0; i < children.length - 1; i++) {
                const current = children[i];
                const next = children[i + 1];

                if (
                    isLikelyContent(current) &&
                    isLikelyHeading(next) &&
                    !(i > 0 && isLikelyHeading(children[i - 1]))
                ) {
                    try {
                        container.insertBefore(next, current);
                    } catch (e) {
                        // fallback in case insertBefore fails
                        const temp = next.cloneNode(true);
                        container.insertBefore(temp, current);
                        next.remove();
                    }
                }
            }
        });
    })();
    """


def get_dialogue_foundry_removal_js():
    """Return JavaScript code that removes Dialogue Foundry app elements from the DOM.

    This script removes elements with ID 'dialogue-foundry-app' which can
    interfere with content extraction.
    """
    return """
    (async () => {
        // Find and remove elements with ID 'dialogue-foundry-app'
        const dialogueElements = document.querySelectorAll('#dialogue-foundry-app');
        
        if (dialogueElements.length > 0) {
            console.log(`Found ${dialogueElements.length} dialogue-foundry elements`);
            
            dialogueElements.forEach(el => {
                el.remove();
            });
        }
    })();
    """


def normalize_url(url):
    """
    Normalize a URL by removing query parameters and fragments.

    Args:
        url (str): The URL to normalize

    Returns:
        str: Normalized URL with only scheme, netloc, and path
    """
    parsed = urlparse(url)
    # Keep only scheme, netloc, and path components
    normalized = urlunparse(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip("/") or "/",  # Keep slash only for root
            "",  # Remove params
            "",  # Remove query
            "",  # Remove fragment
        )
    )
    return normalized


def should_exclude_path(url, excluded_paths):
    """
    Check if a URL should be excluded based on excluded paths.

    Args:
        url (str): The URL to check
        excluded_paths (List[str]): List of path patterns to exclude

    Returns:
        bool: True if URL should be excluded, False otherwise
    """
    if not excluded_paths:
        return False

    for excluded_path in excluded_paths:
        if excluded_path in url:
            return True
    return False


def is_file_url(url):
    """Check if a URL points to a file (has a file extension).

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL points to a file, False otherwise
    """
    # Parse the URL and extract the path
    parsed_url = urlparse(url)
    path = parsed_url.path.lower()

    # Check if the path contains a dot followed by letters/numbers
    # This indicates a file extension
    return "." in path and any(c.isalnum() for c in path.split(".")[-1])


def is_valid_web_url(url):
    """Check if a URL is a valid web link with HTTPS.

    Args:
        url (str): The URL to check

    Returns:
        bool: True if the URL is a valid HTTPS web link, False otherwise
    """
    try:
        parsed_url = urlparse(url)
        return (
            parsed_url.scheme in ("http", "https")
            and parsed_url.netloc  # Ensures there's a domain
            and "." in parsed_url.netloc  # Basic domain validation
        )
    except Exception:
        return False


def extract_links_from_html(html, base_url):
    """Extract internal links directly from HTML when markdown conversion fails.

    Args:
        html (str): The HTML content to extract links from
        base_url (str): The base URL to determine internal vs external links

    Returns:
        list: List of internal link URLs
    """
    import re
    from urllib.parse import urljoin, urlparse

    # Extract all href attributes from anchor tags
    href_pattern = r'<a[^>]*href=["\']([^"\']*)["\'][^>]*>'
    all_links = re.findall(href_pattern, html)

    # Parse base URL to get domain
    base_domain = urlparse(base_url).netloc.lower()

    internal_links = []
    for link in all_links:
        # Skip empty links, fragments, and javascript
        if not link or link.startswith("#") or link.startswith("javascript:"):
            continue

        # Convert relative URLs to absolute
        absolute_url = urljoin(base_url, link)
        parsed_link = urlparse(absolute_url)

        # Check if it's an internal link (same domain)
        if parsed_link.netloc.lower() == base_domain:
            internal_links.append(absolute_url)

    return internal_links
