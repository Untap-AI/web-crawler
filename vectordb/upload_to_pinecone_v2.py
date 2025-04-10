#!/usr/bin/env python3
"""
New Pinecone Upload Script

This script performs two main functions:
1. Upload and embed new content chunks scraped from a client's website
   (generated by the crawler)
2. Delete any previously upserted web-scraped chunks so that only the
   current day's data remains.

All web-scraped records are stored in a dedicated namespace
(e.g. "web_crawl_{CUSTOMER_ID}").
On the Serverless/Starter plan, metadata-based deletion isn't supported
so we delete all vectors in that namespace and then upsert the new data.

Usage:
    python upload_to_pinecone_new.py [--dry-run] [--input-dir DIRECTORY]
"""

import os
import sys
import glob
import json
import argparse
import logging
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
import re
from dotenv import load_dotenv
from pinecone import Pinecone  # Updated import for Pinecone SDK v6+
from pathlib import Path
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def read_markdown_and_metadata(file_path: str) -> Tuple[Dict[str, Any], str]:
    """
    Reads the markdown file content and the companion JSON metadata if available.

    Args:
        file_path: Path to the markdown file.

    Returns:
        A tuple (metadata, content) where metadata is a dictionary (empty if missing)
        and content is the markdown text.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Expect companion metadata file with same name, replacing ".md" with "_metadata.json"
    metadata_path = re.sub(r"\.md$", "_metadata.json", file_path)
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Loaded metadata from {metadata_path}")
        except Exception as e:
            logger.warning(f"Could not load metadata from {metadata_path}: {e}")
    return metadata, content


class PineconeUploader:
    """
    Handles connecting to Pinecone, uploading new records using integrated inference,
    and deleting outdated web-scraped records by wiping the dedicated namespace.
    """

    def __init__(
        self,
        api_key: str,
        environment: str,
        index_name: str,
        customer_id: str,
        index_host: str = None,
    ):
        """
        Initializes the uploader and connects to the correct Pinecone index.
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.customer_id = customer_id
        self.web_namespace = ""  # Use default namespace
        self.batch_id = f"batch_{uuid.uuid4().hex}"

        self.index_host = index_host

        # Initialize the Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Add diagnostic logging
        logger.info(f"Initializing Pinecone client with index: {index_name}")
        logger.info(f"Using environment: {environment}")
        if index_host:
            logger.info(f"Using custom host: {index_host}")

        # Connect to the index
        if self.index_host:
            logger.info(f"Connecting to index using host: {self.index_host}")
            self.index = self.pc.Index(self.index_name, host=self.index_host)
        else:
            try:
                self.index = self.pc.Index(self.index_name)
                logger.info(f"Connected to index: {self.index_name}")
            except Exception as e:
                logger.error(f"Failed to connect to index: {e}")
                raise

        # Verify index connection
        try:
            stats = self.index.describe_index_stats()
            logger.info(f"Index stats: {stats}")
        except Exception as e:
            logger.error(f"Failed to get index stats: {e}")

    def sanitize_vector_id(self, id_str: str) -> str:
        """
        Sanitize vector ID to ensure it contains only ASCII characters.
        """
        normalized = unicodedata.normalize("NFKD", id_str)
        ascii_str = normalized.encode("ASCII", "ignore").decode("ASCII")
        sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", ascii_str)
        return sanitized

    def prepare_records(self, chunks_dir: str) -> List[Dict[str, Any]]:
        """
        Scans the provided directory for markdown chunk files and builds a list of records.
        Each record is tagged with the current batch_id and upload timestamp.
        """
        record_list = []
        md_files = glob.glob(os.path.join(chunks_dir, "*.md"))
        if not md_files:
            logger.warning(f"No markdown files found in directory: {chunks_dir}")
            return []
        current_time = datetime.now().isoformat()
        for md_file in md_files:
            try:
                metadata, content = read_markdown_and_metadata(md_file)
                if "chunk_id" not in metadata:
                    metadata["chunk_id"] = os.path.splitext(os.path.basename(md_file))[
                        0
                    ]
                metadata["batch_id"] = self.batch_id
                metadata["upload_timestamp"] = current_time
                metadata["source_type"] = "web_crawl"
                record = {
                    "id": metadata["chunk_id"],
                    "chunk_text": content,
                    "metadata": metadata,
                }
                record_list.append(record)
                logger.info(f"Prepared record for file: {md_file}")
            except Exception as e:
                logger.error(f"Error processing file {md_file}: {e}")
                continue
        return record_list

    def upsert_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Upserts the provided records to the Pinecone index using server-side embedding.
        Uses the upsert_records method which handles text-to-vector conversion on Pinecone's side.
        """
        try:
            logger.info(
                f"Upserting {len(records)} records into namespace: {self.web_namespace}"
            )

            # Format records for upsert_records
            formatted_records = []
            for record in records:
                metadata = record["metadata"]
                # Create a record with ID and text field for embedding
                formatted_record = {
                    # Provide ID directly as a field
                    "_id": self.sanitize_vector_id(
                        f"web_crawl_{self.batch_id}_{metadata['chunk_id']}"
                    ),
                    # Text to be embedded - assuming "chunk_text" is the field mapped for embedding
                    "chunk_text": record["chunk_text"],
                    # Additional metadata
                    "url": metadata["url"],
                    "title": metadata["title"],
                    "source_domain": metadata.get("source_domain", ""),
                    "source_path": metadata.get("source_path", ""),
                    "keywords": metadata.get("keywords", []),
                    "token_count": metadata.get("token_count", 0),
                    "chunk_id": metadata["chunk_id"],
                    "chunk_index": metadata["chunk_index"],
                    "total_chunks": metadata["total_chunks"],
                    "chunk_name": metadata.get("chunk_name", ""),
                    "crawl_timestamp": metadata.get("crawl_timestamp", ""),
                    "upload_timestamp": metadata["upload_timestamp"],
                    "batch_id": metadata["batch_id"],
                    "source_type": metadata["source_type"],
                    "customer_id": metadata.get("customer_id", ""),
                }
                formatted_records.append(formatted_record)

            # Use upsert_records which handles server-side embedding
            self.index.upsert_records(
                namespace=self.web_namespace, records=formatted_records
            )

            logger.info("Upsert operation completed successfully")
            return len(records)
        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            raise

    def delete_records_by_date_prefix(self, date_str=None) -> int:
        """
        Deletes vectors with ID prefixes matching a specific date.
        If no date is provided, deletes yesterday's vectors.
        This allows for precise deletion of old web crawl data without affecting
        other data in the default namespace.

        Args:
            date_str: Optional date string in YYYYMMDD format. If not provided,
                    defaults to yesterday's date.

        Returns:
            The number of deleted records.
        """
        try:
            # If no date provided, use yesterday's date
            if not date_str:
                from datetime import datetime, timedelta

                yesterday = datetime.now() - timedelta(days=1)
                date_str = yesterday.strftime("%Y%m%d")

            # Create the prefix pattern to match
            prefix = f"web_crawl_{date_str}"
            logger.info(f"Listing vectors with prefix '{prefix}' for deletion")

            # Get all vector IDs matching the prefix
            total_deleted = 0
            deleted_batch = 0

            # Paginate through all vectors with this prefix
            for ids_batch in self.index.list(
                prefix=prefix, namespace=self.web_namespace
            ):
                if ids_batch:
                    logger.info(f"Deleting batch of {len(ids_batch)} vectors")
                    self.index.delete(ids=ids_batch, namespace=self.web_namespace)
                    deleted_batch += len(ids_batch)
                    total_deleted += len(ids_batch)

                    # Log progress for large deletions
                    if deleted_batch >= 10000:
                        logger.info(f"Deleted {total_deleted} vectors so far...")
                        deleted_batch = 0

                    # Small pause to avoid overwhelming the API
                    time.sleep(0.1)

            logger.info(
                f"Successfully deleted {total_deleted} vectors with prefix '{prefix}'"
            )
            return total_deleted

        except Exception as e:
            logger.error(
                f"Error during targeted deletion of vectors with prefix '{date_str}': {e}"
            )
            raise

    def delete_old_web_crawl_records(self, keep_most_recent=1) -> int:
        """
        Find and delete older web crawl data, keeping only the most recent crawls.
        """
        try:
            # Add a small delay to ensure Pinecone has processed the recent upsert
            time.sleep(2)

            # First check if we have any vectors at all
            stats = self.index.describe_index_stats()
            total_vectors = stats["total_vector_count"]
            logger.info(f"Total vectors in index: {total_vectors}")

            if total_vectors == 0:
                logger.info("Index is empty, nothing to delete")
                return 0

            # Get all vectors with web_crawl_ prefix
            all_vectors = []
            logger.info("Listing all web crawl vectors...")
            current_prefix = f"web_crawl_{self.batch_id}"

            # First get vectors NOT matching current batch (these will be deleted)
            for batch in self.index.list(
                prefix="web_crawl_", namespace=self.web_namespace
            ):
                logger.info(f"Found batch of {len(batch)} vectors")
                # Filter out vectors from current batch
                old_vectors = [v for v in batch if not v.startswith(current_prefix)]
                if old_vectors:
                    logger.info(
                        f"Found {len(old_vectors)} vectors from previous batches"
                    )
                    all_vectors.extend(old_vectors)
                else:
                    logger.info("No old vectors found in this batch")

            if not all_vectors:
                logger.info("No old vectors found to delete")
                return 0

            # Delete old vectors in batches
            total_deleted = 0
            batch_size = 100
            for i in range(0, len(all_vectors), batch_size):
                batch = all_vectors[i : i + batch_size]
                logger.info(f"Deleting batch of {len(batch)} vectors: {batch[:5]}...")
                self.index.delete(ids=batch, namespace=self.web_namespace)
                total_deleted += len(batch)
                logger.info(f"Deleted batch of {len(batch)} old vectors")
                time.sleep(0.1)  # Small pause between batches

            logger.info(
                f"Successfully deleted {total_deleted} vectors from old batches"
            )
            return total_deleted

        except Exception as e:
            logger.error(f"Error during deletion of old web crawl records: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Upload and update Pinecone records from web-crawled content chunks."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Process and prepare records, but do not perform any upsert or delete operations",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="crawler/winery_content",
        help="Directory containing the web crawl content (expects a 'chunks' subdirectory)",
    )
    parser.add_argument(
        "--delete-date",
        type=str,
        default="yesterday",
        help="Date to delete in YYYYMMDD format. Use 'yesterday' for yesterday's date, or 'none' to skip deletion",
    )
    parser.add_argument(
        "--keep-recent",
        type=int,
        default=1,
        help="Number of most recent crawls to keep (default: 1). Older crawls will be deleted.",
    )
    args = parser.parse_args()

    load_dotenv()
    # Validate required environment variables.
    required_vars = ["PINECONE_API_KEY", "CUSTOMER_ID", "PINECONE_INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    api_key = os.getenv("PINECONE_API_KEY")
    customer_id = os.getenv("CUSTOMER_ID")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    index_name = os.getenv("PINECONE_INDEX_NAME")
    index_host = os.getenv("PINECONE_HOST")  # optional

    chunks_dir = os.path.join(args.input_dir, "chunks")
    if not os.path.exists(chunks_dir):
        logger.error(f"Chunks directory not found: {chunks_dir}")
        sys.exit(1)

    # Initialize the PineconeUploader instance.
    uploader = PineconeUploader(
        api_key, environment, index_name, customer_id, index_host
    )
    records = uploader.prepare_records(chunks_dir)
    if not records:
        logger.error("No records to process; exiting.")
        sys.exit(1)

    if args.dry_run:
        logger.info(
            "Dry run mode active: records prepared but not uploaded or deleted."
        )
        sys.exit(0)

    # Upsert new records.
    upserted_count = uploader.upsert_records(records)

    # Handle deletion based on command line argument
    if args.delete_date.lower() == "none":
        logger.info("Skipping deletion of old records as requested")
        deleted_count = 0
    else:
        deleted_count = uploader.delete_old_web_crawl_records(args.keep_recent)

    logger.info(
        f"Upload complete: {upserted_count} records upserted, {deleted_count} old records deleted."
    )


if __name__ == "__main__":
    main()
