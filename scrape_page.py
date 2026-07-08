#!/usr/bin/env python3
"""
Shallow, fast scrape entrypoint for the demo-setup pipeline.

Unlike orchestrator.py (which does a deep crawl, chunks, summarizes, and upserts
to the vector store), this fetches the homepage and a few high-signal internal
pages with a real browser (crawl4ai) and prints a JSON payload to stdout:

    {
      "pages": [{"url": "...", "markdown": "..."}, ...],
      "homepageHtml": "<rendered HTML of the homepage>",
      "brand": {"cssVariables": [...], "computedColors": [...],
                "pixelDominantColors": [...], "logos": [...], ...},
      "screenshot": "<base64 JPEG of the homepage's first ~1600px>",
      "screenshotWidth": 1200, "screenshotHeight": 1600
    }

The Node demo-setup service spawns this as a subprocess and parses stdout. All
logs go to stderr so stdout stays pure JSON.

Reuses the crawler's tuned browser settings (real Chrome channel, stealth,
fixed UA) so JS-heavy and anti-bot-protected sites render like the deep crawler.
No LLM content filter is used here — we want the raw rendered markdown/HTML fast.

The `brand` payload comes from brand_probe.js, evaluated inside the settled page.
Reading getComputedStyle beats regexing the returned HTML: external stylesheets
(where modern sites keep all their color) never appear in the HTML at all.
"""

import os
import re
import sys
import json
import base64
import asyncio
from collections import Counter
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawler.config import CrawlerConfig

# Cap the screenshot we hand to the vision model. The nav bar, hero, and primary
# CTA all live above the fold; everything below is footer noise that costs tokens.
SCREENSHOT_MAX_HEIGHT = 1600
SCREENSHOT_MAX_WIDTH = 1200
SCREENSHOT_JPEG_QUALITY = 80

# Paths whose content actually helps a chat demo answer questions, best first.
# Without this we scrape whatever the nav happens to link first, which on most
# sites is a login page and a blog index.
PATH_PRIORITIES = (
    (60, re.compile(r"/(about|about-us|our-story|who-we-are)\b")),
    (55, re.compile(r"/(service|services|what-we-do|solutions|offerings)\b")),
    (55, re.compile(r"/(product|products|shop|menu|collection)\b")),
    (50, re.compile(r"/(pricing|prices|rates|packages|plans)\b")),
    (45, re.compile(r"/(faq|faqs|questions)\b")),
    (40, re.compile(r"/(contact|contact-us|location|locations|hours|visit)\b")),
    (25, re.compile(r"/(team|staff|testimonial|review)\b")),
)
PATH_PENALTIES = (
    (-100, re.compile(r"/(login|signin|sign-in|register|cart|checkout|account)\b")),
    (-80, re.compile(r"/(privacy|terms|cookie|legal|sitemap|accessibility)\b")),
    (-40, re.compile(r"/(blog|news|press|category|tag|author|archive)/.+")),
    (-30, re.compile(r"\.(pdf|jpg|jpeg|png|zip|mp4|webp|svg)$")),
)

PROBE_JS = (Path(__file__).parent / "brand_probe.js").read_text()
LOGO_PROBE_JS = (Path(__file__).parent / "logo_probe.js").read_text()

# A logo candidate must recur on at least this many distinct pages to count —
# one page alone gives no cross-page signal to discriminate it from hero art.
LOGO_MIN_RECURRING_PAGES = 2
# Position tolerance (px) for "roughly the same spot": logos are pinned in a
# sticky/static header, so their vertical position barely moves across pages.
LOGO_MAX_TOP_VARIANCE = 60

# Dominant-color tiling: a flat UI-chrome tile (header bar, button, footer) has
# near-zero pixel spread; a photo/texture tile doesn't. This threshold is on a
# 0-255 per-channel min/max spread within one downscaled 8x8 tile.
DOMINANT_COLOR_TILE_PX = 8
DOMINANT_COLOR_FLATNESS_THRESHOLD = 20
DOMINANT_COLOR_MAX_WIDTH = 240
# Real brand color often lives in the footer or a below-the-fold CTA, not just
# the hero — so the pixel-color scan looks at the whole page, separately from
# the smaller crop sent to vision. Capped so a long-scrolling / infinite-scroll
# page can't blow this up unboundedly.
DOMINANT_COLOR_MAX_PAGE_HEIGHT = 8000


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


def _score_link(href: str) -> int:
    """Rank an internal link by how likely it is to describe the business."""
    path = (urlparse(href).path or "/").lower().rstrip("/") + "/"
    score = 0
    for weight, pattern in PATH_PRIORITIES:
        if pattern.search(path):
            score += weight
            break
    for weight, pattern in PATH_PENALTIES:
        if pattern.search(path):
            score += weight
    # Prefer shallow pages; /services beats /services/commercial/roofing/gutters.
    score -= 5 * max(0, path.count("/") - 2)
    return score


def _select_links(root_result, start_url: str, limit: int) -> list:
    """Pick the `limit` highest-signal internal links off the homepage."""
    internal = (getattr(root_result, "links", {}) or {}).get("internal", [])
    seen = {start_url.rstrip("/")}
    scored = []
    for link in internal:
        href = link.get("href") if isinstance(link, dict) else None
        if not href:
            continue
        href = href.split("#")[0].rstrip("/")
        if not href or href in seen:
            continue
        seen.add(href)
        score = _score_link(href)
        if score <= -30:
            continue
        scored.append((score, href))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [href for _, href in scored[:limit]]


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
        # crawl4ai's default (1080x600) is barely taller than a nav bar — most
        # homepages render their real UI chrome (buttons, footer) below that,
        # and a short viewport also under-triggers scroll-based lazy loading.
        viewport_width=1280,
        viewport_height=1400,
        # crawl4ai prints its own init banner straight to stdout when verbose,
        # which would corrupt the pure-JSON contract this script promises.
        verbose=False,
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
        # Subpages reuse the warmed session and never see the challenge, so they
        # only need enough time for above-the-fold JS to settle.
        delay_before_return_html=(
            config.challenge_wait if is_root else config.subpage_wait
        ),
        magic=config.magic,
        simulate_user=config.simulate_user,
        override_navigator=config.override_navigator,
        user_agent=config.user_agent,
        process_iframes=True,
        remove_overlay_elements=True,
        # Colors/fonts only need the homepage. The logo probe runs on every page
        # (root + subpages) — the dedicated logo hunt below needs candidates from
        # each page to compute cross-page recurrence.
        js_code=[PROBE_JS, LOGO_PROBE_JS] if is_root else [LOGO_PROBE_JS],
        # Each subpage gets its own session so they can run concurrently; the
        # root keeps the shared one it warmed up solving the challenge.
        session_id="scrape_session" if is_root else None,
    )


def _js_results(result) -> list:
    """crawl4ai's js_execution_result envelope:
    {"success": bool, "results": [{"success": bool, "result": <ours>}, ...]}
    one entry per script in the js_code list, in order. Any deviation means a
    script threw or the page blocked evaluation."""
    envelope = getattr(result, "js_execution_result", None) or {}
    return [
        entry["result"]
        for entry in envelope.get("results") or []
        if isinstance(entry, dict) and "result" in entry
    ]


def _extract_probe(result) -> dict:
    """Pull the brand_probe.js object out of the js_execution_result envelope.
    Falls back to {} (Node then falls back to HTML parsing) if the probe threw."""
    for value in _js_results(result):
        if isinstance(value, dict):
            return value
    log(f"Brand probe returned no result (envelope: {str(getattr(result, 'js_execution_result', None))[:200]})")
    return {}


def _extract_logo_candidates(result) -> list:
    """Pull logo_probe.js's candidate array out of the same envelope."""
    for value in _js_results(result):
        if isinstance(value, list):
            return value
    return []


def _group_logo_candidates(candidates_by_page: list, start_url: str) -> dict:
    """The defining property of a logo, independent of any markup convention:
    it's the one image that appears in the same spot on every page, while hero
    photos, product shots, and promos never repeat. Groups per-page candidates
    (from logo_probe.js) by identity so recurrence across distinct pages can be
    scored — this is a *signal*, not a filter, since a homepage-only hero-style
    emblem never gets the chance to recur but can still be the real logo.

    `candidates_by_page` is a list of (page_url, candidate_dict) pairs. Each
    group keeps one representative rect (preferring the homepage occurrence) so
    a recurrence-only find can still participate in vision-bbox matching.
    """
    groups = {}
    for page_url, candidate in candidates_by_page:
        key = candidate["key"]
        group = groups.setdefault(
            key,
            {
                "url": candidate["url"],
                "kind": candidate["kind"],
                "pages": set(),
                "tops": [],
                "widths": [],
                "heights": [],
                "links_home": 0,
                "rect": None,
            },
        )
        group["pages"].add(page_url)
        group["tops"].append(candidate["top"])
        group["widths"].append(candidate["width"])
        group["heights"].append(candidate["height"])
        if candidate.get("linksHome"):
            group["links_home"] += 1
        if group["rect"] is None or page_url == start_url:
            group["rect"] = {
                "top": candidate["top"],
                "left": candidate["left"],
                "width": candidate["width"],
                "height": candidate["height"],
            }
    return groups


def _logo_recurrence_score(group: dict) -> int:
    page_count = len(group["pages"])
    top_variance = max(group["tops"]) - min(group["tops"])
    avg_width = sum(group["widths"]) / len(group["widths"])
    avg_height = sum(group["heights"]) / len(group["heights"])
    ratio = avg_width / max(avg_height, 1)

    score = page_count * 100
    score += 40 if group["links_home"] > 0 else 0
    score += 20 if 1 <= ratio <= 8 else 0  # wordmark proportions
    score -= min(80, top_variance)  # position drift across pages
    return round(score)


def _normalize_img_key(url: str) -> str:
    """Mirrors logo_probe.js's normalizeUrlKey so a homepage img candidate (raw
    URL with resize params) can be matched against recurrence groups (keyed by
    the same normalized form)."""
    try:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    except ValueError:
        base = url
    return "img:" + re.sub(r"~mv2.*$", "", base, flags=re.IGNORECASE).lower()


def _apply_logo_recurrence(
    brand: dict, candidates_by_page: list, start_url: str, page_count: int
) -> None:
    """Cross-check brand_probe.js's homepage candidates (which carry real
    position data, needed for vision-bbox matching) against multi-page
    recurrence, and fold in any recurring candidate the homepage probe missed
    entirely — without discarding position data on the ones it did find."""
    groups = _group_logo_candidates(candidates_by_page, start_url)
    logos = brand.get("logos") or []

    matched_keys = set()
    for logo in logos:
        if logo.get("kind") != "img":
            continue
        key = _normalize_img_key(logo["url"])
        group = groups.get(key)
        if group and len(group["pages"]) >= LOGO_MIN_RECURRING_PAGES:
            matched_keys.add(key)
            recurring_pages = len(group["pages"])
            logo["score"] = logo.get("score", 0) + recurring_pages * 30
            logo["recurringPages"] = recurring_pages

    for key, group in groups.items():
        if key in matched_keys or len(group["pages"]) < LOGO_MIN_RECURRING_PAGES:
            continue
        logos.append(
            {
                "url": group["url"],
                "kind": group["kind"],
                "score": _logo_recurrence_score(group),
                "recurringPages": len(group["pages"]),
                "rect": group["rect"],
            }
        )

    if logos:
        log(
            f"Logo recurrence: checked against {page_count} pages scraped, "
            f"{sum(1 for l in logos if l.get('recurringPages'))} candidate(s) recur"
        )
    brand["logos"] = sorted(logos, key=lambda l: l.get("score", 0), reverse=True)[:10]


def _saturation_lightness(r: int, g: int, b: int) -> tuple:
    mx, mn = max(r, g, b) / 255, min(r, g, b) / 255
    lightness = (mx + mn) / 2
    if mx == mn:
        return 0.0, lightness
    delta = mx - mn
    saturation = delta / (2 - mx - mn) if lightness > 0.5 else delta / (mx + mn)
    return saturation, lightness


def _is_brandworthy_rgb(r: int, g: int, b: int) -> bool:
    """White/near-white and near-black are almost always page background or
    body text, not a chosen brand color — and a page's background is by
    definition its single largest flat region, so without this the dominant-
    color scan would just rediscover "the page is white" every time."""
    saturation, lightness = _saturation_lightness(r, g, b)
    return saturation >= 0.15 and 0.08 <= lightness <= 0.9


def _largest_connected_blob(grid: dict, start: tuple, visited: set) -> int:
    """4-connected flood fill; returns the size (tile count) of the connected
    region of same-bucket tiles containing `start`."""
    if start in visited:
        return 0
    bucket = grid[start]
    stack = [start]
    size = 0
    while stack:
        cell = stack.pop()
        if cell in visited or grid.get(cell) != bucket:
            continue
        visited.add(cell)
        size += 1
        x, y = cell
        stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])
    return size


def _dominant_flat_colors(png_bytes: bytes, max_candidates: int = 8) -> list:
    """What a designer's eye does instinctively: which flat color visually
    dominates the page? DOM sampling (brand_probe.js's computedColors) weighs
    every styled element roughly equally, so a handful of small buttons can
    lose to dozens of plain text links. This instead looks at the actual
    rendered pixels directly.

    A local per-tile variance filter alone isn't enough: a smooth gradient sky
    behind a hero photo has near-zero variance *within* any single small tile,
    so it reads as "flat" too, and a gradient can easily cover more pixels than
    a small solid-colored button. The real discriminator is contiguity — a
    genuine UI element (header bar, button, footer) is one large *contiguous*
    block of near-identical color, while a gradient's matching-quantized tiles
    are scattered along a thin, continuously-drifting band. So: tile the image,
    quantize each flat tile's color, then flood-fill same-color tiles into
    connected blobs and rank by the size of the single largest blob per color —
    not by how many scattered tiles share that color."""
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    if image.width > DOMINANT_COLOR_MAX_WIDTH:
        ratio = DOMINANT_COLOR_MAX_WIDTH / image.width
        image = image.resize(
            (DOMINANT_COLOR_MAX_WIDTH, max(1, int(image.height * ratio))),
            Image.BILINEAR,
        )

    tile = DOMINANT_COLOR_TILE_PX
    grid = {}
    tile_count = 0
    for ty, y in enumerate(range(0, image.height - tile + 1, tile)):
        for tx, x in enumerate(range(0, image.width - tile + 1, tile)):
            extrema = image.crop((x, y, x + tile, y + tile)).getextrema()
            if max(hi - lo for lo, hi in extrema) > DOMINANT_COLOR_FLATNESS_THRESHOLD:
                continue  # textured — not a candidate UI element
            r, g, b = (round((lo + hi) / 2) for lo, hi in extrema)
            if not _is_brandworthy_rgb(r, g, b):
                continue
            # Quantize just enough to group anti-aliased shades of a TRUE flat
            # color together. A coarser bucket would also swallow a *slowly
            # drifting* gradient (a hero photo's sky) into one bucket across a
            # wide contiguous span, letting it pass the blob-size check below
            # even though it's clearly not solid UI chrome. A real flat
            # rectangle has zero drift regardless of bucket size, so tightening
            # this specifically penalizes gradients without hurting real matches.
            grid[(tx, ty)] = (r // 6 * 6, g // 6 * 6, b // 6 * 6)
            tile_count += 1

    if not tile_count:
        return []

    visited: set = set()
    largest_blob = Counter()
    for cell in list(grid.keys()):
        if cell in visited:
            continue
        size = _largest_connected_blob(grid, cell, visited)
        bucket = grid[cell]
        largest_blob[bucket] = max(largest_blob[bucket], size)

    # A genuine UI element needs to span a real chunk of the page, not a couple
    # of stray matching tiles — otherwise noise from anti-aliased edges creeps
    # back in as "the biggest blob" on pages with no solid chrome at all.
    min_blob_tiles = max(4, tile_count // 40)
    ranked = [
        (bucket, size)
        for bucket, size in largest_blob.most_common()
        if size >= min_blob_tiles
    ]

    return [
        {
            "hex": "#{:02x}{:02x}{:02x}".format(*bucket),
            "coverage": round(size / tile_count, 3),
        }
        for bucket, size in ranked[:max_candidates]
    ]


def _shrink_screenshot(png_bytes: bytes) -> dict:
    """Crop to the fold, downscale, JPEG-encode. Returns base64 (no data: prefix)
    plus the final pixel dimensions, so a percentage-based vision bounding box
    can be converted back to page pixel coordinates precisely."""
    image = Image.open(BytesIO(png_bytes)).convert("RGB")
    if image.height > SCREENSHOT_MAX_HEIGHT:
        image = image.crop((0, 0, image.width, SCREENSHOT_MAX_HEIGHT))
    if image.width > SCREENSHOT_MAX_WIDTH:
        ratio = SCREENSHOT_MAX_WIDTH / image.width
        image = image.resize(
            (SCREENSHOT_MAX_WIDTH, int(image.height * ratio)), Image.LANCZOS
        )
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=SCREENSHOT_JPEG_QUALITY)
    return {
        "b64": base64.b64encode(buffer.getvalue()).decode("utf-8"),
        "width": image.width,
        "height": image.height,
    }


def _make_screenshot_hook(state: dict):
    """Captures two different screenshots off the same settled page:
    - A small fold-height crop, shown to the vision model — keeps LLM image
      tokens (and therefore cost/latency) down, and header + hero is where
      logos and nav colors live anyway.
    - A full-page capture used only for the deterministic pixel-color scan,
      which never touches the LLM so its size doesn't cost anything extra.
      Real brand color often concentrates in a footer band or a below-the-fold
      CTA that the fold crop alone would miss entirely.

    Runs in `before_retrieve_html`, which fires after js_code has run and
    scrolled us back to the top, but before the HTML is serialized."""

    async def hook(page, context=None, **kwargs):
        if state.get("screenshot") is not None:
            return page  # subpage fetch; we already have the homepage shot
        try:
            height = await page.evaluate("document.documentElement.scrollHeight")
            width = await page.evaluate("document.documentElement.scrollWidth")
            clip = {
                "x": 0,
                "y": 0,
                "width": min(int(width) or SCREENSHOT_MAX_WIDTH, 1600),
                "height": min(int(height) or SCREENSHOT_MAX_HEIGHT, SCREENSHOT_MAX_HEIGHT),
            }
            raw = await page.screenshot(clip=clip, type="png")
            shrunk = _shrink_screenshot(raw)
            state["screenshot"] = shrunk["b64"]
            state["screenshotWidth"] = shrunk["width"]
            state["screenshotHeight"] = shrunk["height"]
            log(f"Captured screenshot ({len(shrunk['b64'])} b64 chars)")
        except Exception as error:  # noqa: BLE001 - screenshot is best-effort
            log(f"Screenshot failed: {error}")
            state["screenshot"] = ""
            state["pixelColors"] = []
            return page

        try:
            full_page_raw = await page.screenshot(full_page=True, type="png")
            image = Image.open(BytesIO(full_page_raw))
            if image.height > DOMINANT_COLOR_MAX_PAGE_HEIGHT:
                buffer = BytesIO()
                image.crop((0, 0, image.width, DOMINANT_COLOR_MAX_PAGE_HEIGHT)).save(
                    buffer, format="PNG"
                )
                full_page_raw = buffer.getvalue()
            state["pixelColors"] = _dominant_flat_colors(full_page_raw)
            log(f"{len(state['pixelColors'])} dominant flat color(s) from full page")
        except Exception as error:  # noqa: BLE001 - fall back to the fold crop
            log(f"Full-page capture failed ({error}); scanning fold crop only")
            state["pixelColors"] = _dominant_flat_colors(raw)
        return page

    return hook


async def _scrape_one(crawler, config, href):
    """Fetch a single subpage. Never raises — a dead link shouldn't kill the run."""
    try:
        log(f"Scraping internal page: {href}")
        results = _normalize_results(
            await crawler.arun(href, config=_run_config(config, False))
        )
        if results:
            return {
                "url": href,
                "markdown": _markdown_str(results[0]),
                "logoCandidates": _extract_logo_candidates(results[0]),
            }
    except Exception as error:  # noqa: BLE001 - best effort per page
        log(f"Failed to scrape {href}: {error}")
    return None


async def scrape(start_url: str, max_pages: int) -> dict:
    config = CrawlerConfig.from_environment()
    browser_config = _build_browser_config(config)

    state = {"screenshot": None}

    async with AsyncWebCrawler(config=browser_config) as crawler:
        crawler.crawler_strategy.set_hook(
            "before_retrieve_html", _make_screenshot_hook(state)
        )

        log(f"Scraping homepage: {start_url}")
        response = await crawler.arun(start_url, config=_run_config(config, True))
        results = _normalize_results(response)
        if not results:
            raise RuntimeError(f"No result returned for {start_url}")

        root = results[0]
        homepage_html = getattr(root, "html", "") or ""
        brand = _extract_probe(root)
        pages = [{"url": start_url, "markdown": _markdown_str(root)}]
        logo_candidates_by_page = [
            (start_url, candidate) for candidate in _extract_logo_candidates(root)
        ]

        # Fetch the highest-signal internal pages concurrently. Doing this
        # sequentially meant every page paid the full delay_before_return_html
        # in series, which is where the old ~3min homepage-only scrape went.
        if max_pages > 1:
            links = _select_links(root, start_url, max_pages - 1)
            fetched = await asyncio.gather(
                *(_scrape_one(crawler, config, href) for href in links)
            )
            for page in fetched:
                if not page:
                    continue
                pages.append({"url": page["url"], "markdown": page["markdown"]})
                logo_candidates_by_page.extend(
                    (page["url"], candidate)
                    for candidate in page.get("logoCandidates") or []
                )

        # Dedicated logo hunt: cross-check the homepage probe's position-aware
        # candidates against multi-page recurrence (the image/wordmark that
        # repeats across distinct pages in a consistent spot is almost certainly
        # the logo, regardless of markup convention), without discarding the
        # position data a vision-bbox match needs downstream.
        _apply_logo_recurrence(brand, logo_candidates_by_page, start_url, len(pages))
        brand["pixelDominantColors"] = state.get("pixelColors") or []

    return {
        "pages": pages,
        "homepageHtml": homepage_html,
        "brand": brand,
        "screenshot": state.get("screenshot") or "",
        "screenshotWidth": state.get("screenshotWidth") or 0,
        "screenshotHeight": state.get("screenshotHeight") or 0,
    }


def main():
    start_url = sys.argv[1] if len(sys.argv) > 1 else os.getenv("SCRAPE_URL")
    if not start_url:
        log("Usage: python scrape_page.py <url>  (or set SCRAPE_URL)")
        sys.exit(1)

    max_pages = int(os.getenv("SCRAPE_MAX_PAGES", "8"))

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
