"""
Configuration module for web crawler settings.
"""

import os
import yaml
import json
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# override=False: fill in anything not already set in the environment, but
# never clobber vars a caller (e.g. demo-setup's per-company subprocess env)
# explicitly set. override=True previously let a stale .env value silently
# replace the real per-invocation START_URLS/MAX_DEPTH/HEADLESS.
load_dotenv(override=False)


@dataclass
class CrawlerConfig:
    """Configuration class for the web crawler."""

    # Crawling parameters
    start_urls: List[str] = field(default_factory=list)
    max_depth: int = 3
    # Hard ceiling on pages visited by the deep crawl. max_depth alone is not a
    # bound — depth 1 on a site with a 400-link footer is a 400-page crawl.
    max_pages: int = 100
    batch_size: int = 10
    include_external: bool = False

    # Content filtering
    excluded_tags: List[str] = field(
        default_factory=lambda: ["footer", "nav", "header"]
    )
    excluded_paths: List[str] = field(default_factory=list)
    exclude_hidden_elements: bool = True
    delay_before_return_html: int = 5
    # Extra wait (seconds) on the first/root fetch so JS anti-bot challenges
    # (e.g. SiteGround's "Robot Challenge Screen") can resolve and reload.
    challenge_wait: int = 12
    # Wait (seconds) on subsequent pages in the shallow demo scrape. The anti-bot
    # challenge only fires on first load, and these are fetched concurrently, so
    # they need far less than the root.
    subpage_wait: int = 1

    # Validation parameters
    expected_chunks: int = 0  # 0 means no validation
    chunk_threshold_pct: float = 20.0  # Default 20% threshold

    # LLM settings
    llm_provider: str = "openai/gpt-4.1-nano"
    llm_instruction: str = """
        You are an assistant who is an expert at filtering content extracted from websites. You are given a page from a website.
        Your task is to extract ONLY substantive content that provides real value to customers visiting the website.
        The purpose of the content is to help customers learn about the company and its products by using it as chunks for RAG.
        
        IMPORTANT: Do NOT generate or create any content that doesn't explicitly exist on the page. Only extract content that is actually present.
        
        FORMAT REQUIREMENTS:
        - Use clear, hierarchical headers (H1, H2, H3) for each section
        - Add new lines between paragraphs and lists for better readability
        - Create concise, scannable bulleted lists for important details
        - Organize content logically while maintaining its actual presence on the page
        - Preserve exact pricing, dates, hours, and contact information
        - Preserve the existing content structure - do not invent new organization
        
        Exclude:
        - ALL navigation links and menu items
        - ALL social media links and sharing buttons
        - Any other information that is not relevant to the company and its products
        - ALL images
        
        Remember: Only include content that actually exists on the page. Do not invent, generate, or create any content that isn't present.
    """
    chunk_token_threshold: int = 400
    overlap_rate: float = 0.2
    # Per-request ceiling (seconds) on the content-filter LLM call. Passed
    # through to litellm.completion via LLMContentFilter's extra_args.
    llm_timeout: int = 30

    # Chunking parameters
    embedding_model_name: str = "text-embedding-3-small"
    buffer_size: int = 3

    # Summarization parameters
    summary_model_name: str = "gpt-4.1-nano"
    summary_temperature: float = 0.3
    # gpt-4.1-nano, I/O-bound. At depth 1 x 100 pages this is ~500 calls, and
    # 10-wide made summarization the second-biggest chunk of wall clock.
    summary_max_workers: int = 24

    # Vector DB parameters
    chunk_id_prefix: str = "web_crawl"
    record_retention_hours: int = 1
    upsert_batch_size: int = 50
    delete_old_records: bool = True
    upstash_namespace: str = None

    # Processing control
    dry_run: bool = False
    verbose: bool = False

    # Browser configuration
    browser_type: str = "chromium"
    # Launch a real Chrome channel ("chrome") rather than bundled Chromium —
    # passes JS anti-bot challenges far more reliably. Set to "chromium" on
    # headless servers (e.g. Render) that don't have Chrome installed.
    browser_channel: str = "chrome"
    headless: bool = False
    # light_mode / text_mode strip browser features and produce a bot-like
    # fingerprint that anti-bot challenges flag — keep them off for crawling.
    light_mode: bool = False
    text_mode: bool = False
    ignore_https_errors: bool = True
    # Anti-bot / stealth. magic enables crawl4ai's evasion bundle; the other
    # two patch navigator props and emulate human interaction.
    magic: bool = True
    simulate_user: bool = True
    override_navigator: bool = True
    # A SINGLE fixed User-Agent for the whole session. With magic=True crawl4ai
    # otherwise generates a NEW random UA on every request, which breaks anti-bot
    # clearance cookies (SiteGround binds clearance to the UA) and causes
    # intermittent 403s. Keep it consistent. Override per-environment via
    # USER_AGENT env (e.g. a Linux Chrome UA on a Linux server).
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36"
    )

    # Output directories
    output_dir: str = "cleaned_output"
    logs_dir: str = "logs"

    @classmethod
    def from_environment(cls) -> "CrawlerConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Process START_URLS (comma-separated string to list)
        if "START_URLS" in os.environ:
            config.start_urls = [
                url.strip() for url in os.environ["START_URLS"].split(",")
            ]

        # Process numeric values
        if "MAX_DEPTH" in os.environ:
            config.max_depth = int(os.environ["MAX_DEPTH"])

        if "MAX_PAGES" in os.environ:
            config.max_pages = int(os.environ["MAX_PAGES"])

        if "SUBPAGE_WAIT" in os.environ:
            config.subpage_wait = int(os.environ["SUBPAGE_WAIT"])

        if "BATCH_SIZE" in os.environ:
            config.batch_size = int(os.environ["BATCH_SIZE"])

        # Validation parameters
        if "EXPECTED_CHUNKS" in os.environ:
            config.expected_chunks = int(os.environ["EXPECTED_CHUNKS"])

        if "CHUNK_THRESHOLD_PCT" in os.environ:
            config.chunk_threshold_pct = float(os.environ["CHUNK_THRESHOLD_PCT"])

        if "DELAY_BEFORE_RETURN_HTML" in os.environ:
            config.delay_before_return_html = int(
                os.environ["DELAY_BEFORE_RETURN_HTML"]
            )

        if "EMBEDDING_MODEL_NAME" in os.environ:
            config.embedding_model_name = os.environ["EMBEDDING_MODEL_NAME"]

        if "BUFFER_SIZE" in os.environ:
            config.buffer_size = int(os.environ["BUFFER_SIZE"])

        # Summarization parameters
        if "SUMMARY_MODEL_NAME" in os.environ:
            config.summary_model_name = os.environ["SUMMARY_MODEL_NAME"]

        if "SUMMARY_TEMPERATURE" in os.environ:
            config.summary_temperature = float(os.environ["SUMMARY_TEMPERATURE"])

        if "SUMMARY_MAX_TOKENS" in os.environ:
            config.summary_max_tokens = int(os.environ["SUMMARY_MAX_TOKENS"])

        if "SUMMARY_MAX_WORKERS" in os.environ:
            config.summary_max_workers = int(os.environ["SUMMARY_MAX_WORKERS"])

        # Vector DB parameters
        if "CHUNK_ID_PREFIX" in os.environ:
            config.chunk_id_prefix = os.environ["CHUNK_ID_PREFIX"]

        if "RECORD_RETENTION_HOURS" in os.environ:
            config.record_retention_hours = int(os.environ["RECORD_RETENTION_HOURS"])

        if "UPSERT_BATCH_SIZE" in os.environ:
            config.upsert_batch_size = int(os.environ["UPSERT_BATCH_SIZE"])

        if "DELETE_OLD_RECORDS" in os.environ:
            config.delete_old_records = os.environ["DELETE_OLD_RECORDS"].lower() in [
                "true",
                "1",
                "yes",
            ]

        if "UPSTASH_NAMESPACE" in os.environ:
            config.upstash_namespace = os.environ["UPSTASH_NAMESPACE"]

        if "DRY_RUN" in os.environ:
            config.dry_run = os.environ["DRY_RUN"].lower() in ["true", "1", "yes"]

        if "VERBOSE" in os.environ:
            config.verbose = os.environ["VERBOSE"].lower() in ["true", "1", "yes"]

        # Process lists
        if "EXCLUDED_TAGS" in os.environ:
            try:
                config.excluded_tags = json.loads(os.environ["EXCLUDED_TAGS"])
            except json.JSONDecodeError:
                # Fallback to comma-separated string
                config.excluded_tags = [
                    tag.strip() for tag in os.environ["EXCLUDED_TAGS"].split(",")
                ]

        if "EXCLUDED_PATHS" in os.environ:
            try:
                config.excluded_paths = json.loads(os.environ["EXCLUDED_PATHS"])
            except json.JSONDecodeError:
                # Fallback to comma-separated string
                config.excluded_paths = [
                    path.strip() for path in os.environ["EXCLUDED_PATHS"].split(",")
                ]

        # Directory settings
        if "OUTPUT_DIR" in os.environ:
            config.output_dir = os.environ["OUTPUT_DIR"]

        if "LOGS_DIR" in os.environ:
            config.logs_dir = os.environ["LOGS_DIR"]

        # LLM settings
        if "LLM_PROVIDER" in os.environ:
            config.llm_provider = os.environ["LLM_PROVIDER"]

        if "LLM_INSTRUCTION" in os.environ:
            config.llm_instruction = os.environ["LLM_INSTRUCTION"]

        if "LLM_TIMEOUT" in os.environ:
            config.llm_timeout = int(os.environ["LLM_TIMEOUT"])

        if "EXCLUDE_HIDDEN_ELEMENTS" in os.environ:
            config.exclude_hidden_elements = os.environ[
                "EXCLUDE_HIDDEN_ELEMENTS"
            ].lower() in ["true", "1", "yes"]

        # Browser configuration
        if "BROWSER_TYPE" in os.environ:
            config.browser_type = os.environ["BROWSER_TYPE"]

        if "BROWSER_CHANNEL" in os.environ:
            config.browser_channel = os.environ["BROWSER_CHANNEL"]

        if "CHALLENGE_WAIT" in os.environ:
            config.challenge_wait = int(os.environ["CHALLENGE_WAIT"])

        if "MAGIC" in os.environ:
            config.magic = os.environ["MAGIC"].lower() in ["true", "1", "yes"]

        if "SIMULATE_USER" in os.environ:
            config.simulate_user = os.environ["SIMULATE_USER"].lower() in [
                "true",
                "1",
                "yes",
            ]

        if "OVERRIDE_NAVIGATOR" in os.environ:
            config.override_navigator = os.environ["OVERRIDE_NAVIGATOR"].lower() in [
                "true",
                "1",
                "yes",
            ]

        if "USER_AGENT" in os.environ:
            config.user_agent = os.environ["USER_AGENT"]

        if "HEADLESS" in os.environ:
            config.headless = os.environ["HEADLESS"].lower() in ["true", "1", "yes"]

        if "LIGHT_MODE" in os.environ:
            config.light_mode = os.environ["LIGHT_MODE"].lower() in ["true", "1", "yes"]

        if "TEXT_MODE" in os.environ:
            config.text_mode = os.environ["TEXT_MODE"].lower() in ["true", "1", "yes"]

        if "IGNORE_HTTPS_ERRORS" in os.environ:
            config.ignore_https_errors = os.environ["IGNORE_HTTPS_ERRORS"].lower() in [
                "true",
                "1",
                "yes",
            ]

        return config

    @classmethod
    def from_yaml(cls, yaml_file: str) -> "CrawlerConfig":
        """Load configuration from a YAML file."""
        if not os.path.exists(yaml_file):
            raise FileNotFoundError(f"Config file not found: {yaml_file}")

        with open(yaml_file, "r") as f:
            config_data = yaml.safe_load(f)

        return cls(**config_data)
