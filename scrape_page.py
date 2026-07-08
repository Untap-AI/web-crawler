#!/usr/bin/env python3
"""
Shallow, fast scrape entrypoint for the demo-setup pipeline.

Unlike orchestrator.py (which does a deep crawl, chunks, summarizes, and upserts
to the vector store), this fetches just the homepage and a few internal pages with
a real browser (crawl4ai) and prints a JSON payload to stdout:

    {
      "pages": [{"url": "...", "markdown": "..."}, ...],
      "homepageHtml": "<rendered HTML of the homepage>"
    }

The Node demo-setup service spawns this as a subprocess and parses stdout. All
logs go to stderr so stdout stays pure JSON.

Reuses the crawler's tuned browser settings (real Chrome channel, stealth,
fixed UA) so JS-heavy and anti-bot-protected sites render like the deep crawler.
No LLM content filter is used here — we want the raw rendered markdown/HTML fast.
"""

import os
import sys
import json
import asyncio

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawler.config import CrawlerConfig


def log(message):
    """Write progress to stderr so stdout stays pure JSON."""
    print(message, file=sys.stderr, flush=True)


def _markdown_str(result):
    """crawl4ai markdown can be a str or a MarkdownGenerationResult."""
    markdown = getattr(result, "markdown", "") or ""
    raw = getattr(markdown, "raw_markdown", None)
    return raw if raw is not None else str(markdown)


def _normalize_results(response):
    """arun may return a single result or an iterable/container of results."""
    if response is None:
        return []
    if hasattr(response, "markdown") or hasattr(response, "html"):
        return [response]
    try:
        flat = []
        for item in response:
            if isinstance(item, (list, tuple)):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    except TypeError:
        return [response]


def _build_browser_config(config: CrawlerConfig) -> BrowserConfig:
    browser_config = BrowserConfig(
        browser_type=config.browser_type,
        chrome_channel=config.browser_channel,
        channel=config.browser_channel,
        headless=config.headless,
        light_mode=config.light_mode,
        text_mode=config.text_mode,
        ignore_https_errors=config.ignore_https_errors,
        user_agent=config.user_agent,
    )
    # Same fix as crawl.py: clear chrome_channel so Playwright uses managed
    # chromium instead of falling back to a system Chrome lookup.
    browser_config.chrome_channel = ""
    return browser_config


def _run_config(config: CrawlerConfig, is_root: bool) -> CrawlerRunConfig:
    return CrawlerRunConfig(
        # Default markdown generator (no LLM filter) — fast and raw.
        markdown_generator=None,
        excluded_tags=config.excluded_tags,
        exclude_external_links=True,
        exclude_social_media_links=True,
        exclude_external_images=True,
        verbose=config.verbose,
        # Longer wait on the root fetch so a JS anti-bot challenge can solve.
        delay_before_return_html=(
            config.challenge_wait if is_root else config.delay_before_return_html
        ),
        magic=config.magic,
        simulate_user=config.simulate_user,
        override_navigator=config.override_navigator,
        user_agent=config.user_agent,
        process_iframes=True,
        remove_overlay_elements=True,
        session_id="scrape_session",
    )


async def scrape(start_url: str, max_pages: int) -> dict:
    config = CrawlerConfig.from_environment()
    browser_config = _build_browser_config(config)

    pages = []
    homepage_html = ""

    async with AsyncWebCrawler(config=browser_config) as crawler:
        log(f"Scraping homepage: {start_url}")
        response = await crawler.arun(start_url, config=_run_config(config, True))
        results = _normalize_results(response)
        if not results:
            raise RuntimeError(f"No result returned for {start_url}")

        root = results[0]
        homepage_html = getattr(root, "html", "") or ""
        pages.append({"url": start_url, "markdown": _markdown_str(root)})

        # Optionally scrape a few internal links for richer content analysis.
        if max_pages > 1:
            internal = (getattr(root, "links", {}) or {}).get("internal", [])
            seen = {start_url}
            for link in internal:
                href = link.get("href") if isinstance(link, dict) else None
                if not href or href in seen:
                    continue
                seen.add(href)
                log(f"Scraping internal page: {href}")
                try:
                    sub = _normalize_results(
                        await crawler.arun(href, config=_run_config(config, False))
                    )
                    if sub:
                        pages.append(
                            {"url": href, "markdown": _markdown_str(sub[0])}
                        )
                except Exception as error:  # noqa: BLE001 - best effort per page
                    log(f"Failed to scrape {href}: {error}")
                if len(pages) >= max_pages:
                    break

    return {"pages": pages, "homepageHtml": homepage_html}


def main():
    start_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SCRAPE_URL")
    if not start_url:
        log("Usage: python scrape_page.py <url>  (or set SCRAPE_URL)")
        sys.exit(1)

    max_pages = int(os.getenv("SCRAPE_MAX_PAGES", "5"))

    try:
        payload = asyncio.run(scrape(start_url, max_pages))
    except Exception as error:  # noqa: BLE001 - surface as non-zero exit
        log(f"Scrape failed: {error}")
        sys.exit(1)

    # Pure JSON on stdout for the Node caller.
    json.dump(payload, sys.stdout)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
