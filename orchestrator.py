import sys
import logging
import asyncio
import argparse
import time
import os
from pathlib import Path
from datetime import datetime
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, Content
from crawler.crawl import crawl
from chunk_content import chunk_content
from summary import summarize_content
from vectordb.upload import upload_chunks
from crawler.config import CrawlerConfig

# A piped (non-tty) stdout is block-buffered by default — plain print()
# calls throughout this codebase could sit unflushed for 10+ minutes during
# a slow crawl, hiding real progress from whatever spawned this process
# (demo-setup pipes stdout straight into its own logger).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(
            f'logs/orchestrator_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def format_time(seconds):
    """Format seconds into a human-readable time string."""
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"


async def main(dry_run=False):
    """Main orchestrator function that runs crawler, chunking, summarization,
    and vector store upload in sequence."""
    try:
        # Start timing the entire process
        total_start_time = time.time()

        # Create configuration from environment variables
        config = CrawlerConfig.from_environment()

        # Override dry_run from command line if specified
        if dry_run:
            config.dry_run = True

        logger.info("Starting orchestration process...")
        logger.info(f"Using configuration with {len(config.start_urls)} start URLs")

        # Run crawler with configuration
        logger.info("Starting crawler...")
        crawl_start_time = time.time()
        crawling_results = await crawl(config)
        crawl_time = time.time() - crawl_start_time
        logger.info(
            f"Crawling completed with {len(crawling_results)} results "
            f"in {format_time(crawl_time)}"
        )

        # Run content chunking on the crawl results with configuration
        logger.info("Starting content chunking...")
        chunk_start_time = time.time()
        chunked_results = chunk_content(crawling_results, config)
        chunk_time = time.time() - chunk_start_time
        logger.info(
            f"Chunking completed with {len(chunked_results)} chunks "
            f"in {format_time(chunk_time)}"
        )

        # Run summarization with the chunk results and configuration
        logger.info("Starting content summarization...")
        summary_start_time = time.time()
        summarized_results = summarize_content(chunked_results, config)
        summary_time = time.time() - summary_start_time
        results_count = len(summarized_results)
        logger.info(
            f"Summarization completed with {results_count} results "
            f"in {format_time(summary_time)}"
        )

        if config.dry_run:
            # Save the summarized results instead of uploading to the vector store
            logger.info(f"Saving results to {config.output_dir} folder")
            save_start_time = time.time()
            save_results_to_folder(summarized_results, config.output_dir)
            save_time = time.time() - save_start_time
            logger.info(f"Results saved to folder in {format_time(save_time)}")
        else:
            # Check if the number of chunks to upload meets the expected threshold
            should_upload = True
            upload_time = 0

            # Validate chunk count if expected_chunks is configured
            if config.expected_chunks > 0:
                min_threshold = (
                    config.expected_chunks * (100 - config.chunk_threshold_pct) / 100
                )

                if results_count < min_threshold:
                    threshold_pct = 100 - config.chunk_threshold_pct
                    logger.warning(
                        f"Number of chunks ({results_count}) is below the "
                        f"minimum threshold ({min_threshold:.0f}, "
                        f"{threshold_pct:.0f}% of expected "
                        f"{config.expected_chunks})"
                    )
                    should_upload = False
                    threshold_pct = 100 - config.chunk_threshold_pct
                    email_subject = "Web Crawler Warning: Below Expected Chunks"
                    email_message = (
                        f"The web crawler processed only {results_count} "
                        f"chunks, which is below the minimum threshold of "
                        f"{min_threshold:.0f} chunks ({threshold_pct:.0f}% of "
                        f"expected {config.expected_chunks}).\n\n"
                        f"Upload has been skipped. Please review the "
                        f"crawler configuration and data sources."
                    )
                    send_email_notification(email_subject, email_message)
                elif results_count > config.expected_chunks * (
                    1 + config.chunk_threshold_pct / 100
                ):
                    logger.info(
                        f"Number of chunks ({results_count}) exceeds expected "
                        f"count ({config.expected_chunks})"
                    )
                    # Still upload but send notification
                    email_subject = "Web Crawler Notice: Above Expected Chunks"
                    email_message = (
                        f"The web crawler processed {results_count} chunks, "
                        f"which exceeds the expected count of "
                        f"{config.expected_chunks}.\n\n"
                        f"Upload will proceed normally, but you may "
                        f"want to review your crawler configuration to ensure "
                        f"it's working as intended."
                    )
                    send_email_notification(email_subject, email_message)

            if should_upload:
                upload_start_time = time.time()
                upload_chunks(summarized_results, config)
                upload_time = time.time() - upload_start_time
                logger.info(
                    f"Upload completed in {format_time(upload_time)}"
                )
            else:
                logger.warning("Upload skipped due to validation failure")

        # Calculate and log the total execution time
        total_time = time.time() - total_start_time
        logger.info(
            f"Orchestration completed successfully in {format_time(total_time)}"
        )

        # Print timing summary
        logger.info("===== Performance Summary =====")
        logger.info(f"Crawling:      {format_time(crawl_time)}")
        logger.info(f"Chunking:      {format_time(chunk_time)}")
        logger.info(f"Summarization: {format_time(summary_time)}")
        if config.dry_run:
            logger.info(f"Saving:        {format_time(save_time)}")
        elif should_upload:
            logger.info(f"Upload:        {format_time(upload_time)}")
        else:
            logger.info("Upload:        Skipped")
        logger.info(f"Total time:    {format_time(total_time)}")
        logger.info("=============================")

    except Exception as e:
        # Calculate execution time even if there's an error
        total_time = time.time() - total_start_time
        logger.error(f"Orchestration failed after {format_time(total_time)}: {str(e)}")

        try:
            error_message = (
                f"The web crawler orchestration process failed after "
                f"{format_time(total_time)}.\n\n"
                f"Error: {str(e)}"
            )
            send_email_notification("Web Crawler Error: Process Failed", error_message)
        except Exception as email_error:
            logger.error(f"Failed to send error notification email: {email_error}")

        sys.exit(1)


def save_results_to_folder(results, output_dir="cleaned_output"):
    """Save the chunked results to a folder as simple text files.

    Args:
        results (list): List of processed chunk results
        output_dir (str): Directory to save results to
    """
    try:
        # Create the output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Clear all existing files in the output directory
        logger.info(f"Clearing existing files in {output_dir}")
        existing_files = list(output_dir.glob("*.md"))
        if existing_files:
            for file in existing_files:
                try:
                    file.unlink()
                except Exception as e:
                    logger.warning(f"Failed to delete file {file}: {e}")
            logger.info(f"Removed {len(existing_files)} existing files")
        else:
            logger.info("No existing files found to clear")

        print(f"Saving {len(results)} chunks to {output_dir}")

        # Save each result as a simple text file
        for i, result in enumerate(results):
            # Get URL for filename
            page_path = getattr(result, "chunk_name", "unknown")
            page_path_part = (
                page_path.split("//")[-1].replace("/", "_").replace(":", "_")
            )

            # Create filename with index for uniqueness
            filename = f"{page_path_part}.md"
            file_path = output_dir / filename

            # Write the content to the file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(result.markdown)

            logger.info(f"Saved chunk {i+1} to {file_path}")

    except Exception as e:
        logger.error(f"Error saving chunks to folder: {str(e)}")
        raise


def send_email_notification(subject, message):
    """Send email notification using SendGrid API.

    Args:
        subject (str): Email subject
        message (str): Email message body
    """
    try:
        # Check if SendGrid API key is configured
        api_key = os.environ.get("SENDGRID_API_KEY")
        if not api_key:
            logger.error("SendGrid API key not found in environment variables")
            return

        # Get email sender and recipient from environment
        from_email = os.environ.get("NOTIFICATION_EMAIL_FROM", "no-reply@untap-ai.com")
        to_email = os.environ.get("NOTIFICATION_EMAIL_TO", "contact@untap-ai.com")

        # Create email message
        email = Mail(
            from_email=Email(from_email),
            to_emails=to_email,
            subject=subject,
            plain_text_content=Content("text/plain", message),
        )

        # Send email
        sg = SendGridAPIClient(api_key=api_key)
        response = sg.send(email)

        logger.info(f"Email sent with status: {response.status_code}")

    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run the web crawler orchestration process"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Save results to folder instead of uploading to the vector store",
    )
    parser.add_argument(
        "--config",
        help="Path to YAML configuration file",
    )
    args = parser.parse_args()

    # Load config from file if specified
    config = None
    if args.config:
        try:
            config = CrawlerConfig.from_yaml(args.config)
            if args.dry_run:
                config.dry_run = True
        except Exception as e:
            logger.error(f"Error loading config file: {e}")
            sys.exit(1)

    asyncio.run(main(dry_run=args.dry_run if not config else config.dry_run))
