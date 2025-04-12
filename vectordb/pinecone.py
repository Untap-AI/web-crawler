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
import logging
import time
from datetime import datetime, timedelta
import re
from dotenv import load_dotenv

# Import from the pinecone.control module where the Pinecone class is defined
from pinecone import Pinecone
import unicodedata

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get configuration from environment variables
CHUNK_ID_PREFIX = os.getenv("CHUNK_ID_PREFIX", "web_crawl")
RECORD_RETENTION_HOURS = int(os.getenv("RECORD_RETENTION_HOURS", "1"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "50"))
DELETE_OLD_RECORDS = os.getenv("DELETE_OLD_RECORDS", "true").lower() in [
    "true",
    "1",
    "yes",
]


class PineconeUploader:
    """
    Handles connecting to Pinecone, uploading new records using integrated
    inference, and deleting outdated web-scraped records by wiping the
    dedicated namespace.
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        chunk_id_prefix=None,
        record_retention_hours=None,
        upsert_batch_size=None,
        delete_old_records=None,
    ):
        """
        Initializes the uploader and connects to the correct Pinecone index.

        Args:
            api_key: Pinecone API key
            index_name: Name of Pinecone index
            chunk_id_prefix: Prefix for chunk IDs
            record_retention_hours: How many hours to keep old records before deletion
            upsert_batch_size: Number of records to upsert in each batch
            delete_old_records: Whether to delete old records
        """
        self.api_key = api_key
        self.index_name = index_name
        self.web_namespace = ""  # Use default namespace

        # Use provided values or fall back to environment defaults
        self.chunk_id_prefix = chunk_id_prefix or CHUNK_ID_PREFIX
        self.record_retention_hours = record_retention_hours or RECORD_RETENTION_HOURS
        self.upsert_batch_size = upsert_batch_size or UPSERT_BATCH_SIZE
        self.delete_old_records = (
            delete_old_records if delete_old_records is not None else DELETE_OLD_RECORDS
        )

        # Generate a batch ID using timestamp for tracking current batch
        self.batch_id = datetime.now().strftime("%Y%m%d%H%M%S")

        # Initialize the Pinecone client
        self.pc = Pinecone(api_key=self.api_key)

        # Add diagnostic logging
        logger.info(f"Initializing Pinecone client with index: {index_name}")

        # Log configuration
        logger.info(f"Using chunk ID prefix: {self.chunk_id_prefix}")
        logger.info(f"Record retention hours: {self.record_retention_hours}")
        logger.info(f"Upsert batch size: {self.upsert_batch_size}")
        logger.info(f"Delete old records: {self.delete_old_records}")

        # Connect to the index
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

    def upsert_records(self, records) -> int:
        """
        Upserts the provided records to the Pinecone index using server-side
        embedding. Uses the upsert_records method which handles text-to-vector
        conversion on Pinecone's side.

        Records are processed in batches to prevent timeouts and memory issues.
        """
        try:
            total_records = len(records)
            logger.info(
                f"Upserting {total_records} records into namespace: {self.web_namespace}"
            )

            # Format all records first
            formatted_records = []
            for record in records:
                # Access attributes directly since record is a namespace object
                url = record.url
                chunk_name = record.chunk_name
                chunk_text = record.markdown

                # Create a record with ID and text field for embedding
                formatted_record = {
                    # Provide ID directly as a field
                    "_id": self.sanitize_vector_id(
                        f"{self.chunk_id_prefix}_{chunk_name}"
                    ),
                    # Text to be embedded
                    "chunk_text": chunk_text,
                    # Additional metadata
                    "url": url,
                    "upload_timestamp": datetime.now().isoformat(),
                }
                formatted_records.append(formatted_record)

            # Process records in batches
            batch_size = self.upsert_batch_size
            total_batches = (len(formatted_records) + batch_size - 1) // batch_size
            records_upserted = 0

            logger.info(
                f"Processing {total_batches} batches of up to {batch_size} records each"
            )

            for i in range(total_batches):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, len(formatted_records))
                batch = formatted_records[start_idx:end_idx]

                batch_count = len(batch)
                logger.info(
                    f"Upserting batch {i+1}/{total_batches} with {batch_count} records"
                )

                # Use upsert_records which handles server-side embedding
                self.index.upsert_records(self.web_namespace, batch)
                records_upserted += batch_count

                logger.info(
                    f"Completed batch {i+1}/{total_batches}. "
                    f"Progress: {records_upserted}/{total_records}"
                )

                # Small pause between batches to avoid rate limiting
                if i < total_batches - 1:
                    time.sleep(0.5)

            logger.info(
                f"Upsert operation completed. "
                f"Total records upserted: {records_upserted}"
            )
            return records_upserted
        except Exception as e:
            logger.error(f"Error during upsert: {e}")
            raise

    def delete_older_than_retention_period(self) -> int:
        """
        Delete all records where:
        1. The ID starts with CHUNK_ID_PREFIX
        2. They are older than the retention period based on timestamp metadata

        Returns:
            The number of deleted records.
        """
        if not self.delete_old_records:
            logger.info("Record deletion is disabled, skipping.")
            return 0

        try:
            logger.info(
                f"Starting deletion of records older than "
                f"{self.record_retention_hours} hours"
            )

            # Calculate timestamp for the retention period
            retention_threshold = datetime.now() - timedelta(
                hours=self.record_retention_hours
            )
            logger.info(f"Threshold time: {retention_threshold.isoformat()}")

            # Get all vectors with the CHUNK_ID_PREFIX
            all_old_vectors = []
            logger.info(f"Listing all vectors with prefix '{self.chunk_id_prefix}'...")

            # Get all vector IDs with CHUNK_ID_PREFIX first
            all_matching_ids = []
            for batch in self.index.list(
                prefix=self.chunk_id_prefix, namespace=self.web_namespace
            ):
                if batch:
                    all_matching_ids.extend(batch)
                    logger.info(f"Found batch of {len(batch)} vectors with prefix")
                    # Small pause between batch processing
                    time.sleep(0.1)

            if not all_matching_ids:
                logger.info(f"No vectors with prefix '{self.chunk_id_prefix}' found")
                return 0

            logger.info(f"Total {len(all_matching_ids)} vectors with matching prefix")

            # Fetch all vectors at once to check timestamps
            try:
                logger.info(
                    f"Fetching all {len(all_matching_ids)} vectors to check timestamps"
                )
                # Fetch vectors using the Pinecone v6 format
                response = self.index.fetch(
                    ids=all_matching_ids, namespace=self.web_namespace
                )

                # Access the vectors attribute from the response object
                # In Pinecone v6, response is an object with a 'vectors' attribute
                vectors_dict = response.vectors

                # Process all vectors
                for vector_id, vector_data in vectors_dict.items():
                    upload_timestamp = vector_data.metadata["upload_timestamp"]

                    if upload_timestamp:
                        # Parse timestamp and compare
                        try:
                            record_timestamp = datetime.fromisoformat(upload_timestamp)
                            if record_timestamp < retention_threshold:
                                all_old_vectors.append(vector_id)
                                logger.debug(
                                    f"Vector {vector_id} is older than "
                                    f"{self.record_retention_hours} hours "
                                    f"(uploaded at {upload_timestamp})"
                                )
                        except (ValueError, TypeError) as e:
                            logger.warning(
                                f"Could not parse timestamp {upload_timestamp} "
                                f"for vector {vector_id}: {e}"
                            )
                    else:
                        logger.warning(
                            f"Vector {vector_id} has no upload_timestamp metadata"
                        )

            except Exception as e:
                logger.error(f"Error fetching vectors: {e}")
                import traceback

                traceback.print_exc()
                return 0

            # Check if we found any old vectors
            if not all_old_vectors:
                logger.info(
                    f"No vectors older than {self.record_retention_hours} hours found"
                )
                return 0

            logger.info(
                f"Found {len(all_old_vectors)} vectors older than {self.record_retention_hours} hours"
            )

            # Delete all old vectors at once
            try:
                logger.info(f"Deleting all {len(all_old_vectors)} old vectors at once")
                self.index.delete(ids=all_old_vectors, namespace=self.web_namespace)
                total_deleted = len(all_old_vectors)
                logger.info(
                    f"Successfully deleted {total_deleted} vectors older than "
                    f"{self.record_retention_hours} hours"
                )
                return total_deleted
            except Exception as e:
                logger.error(f"Error deleting vectors: {e}")
                return 0

        except Exception as e:
            logger.error(f"Error during deletion of old records: {e}")
            raise


def upload_chunks(chunks, config=None):
    """
    Upload chunks to Pinecone vector database.

    Args:
        chunks: List of chunks to upload
        config: Optional configuration object
    """
    load_dotenv()
    # Validate required environment variables.
    required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing environment variables: {', '.join(missing)}")
        sys.exit(1)

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME")

    # Get configuration from config object if provided
    chunk_id_prefix = None
    record_retention_hours = None
    upsert_batch_size = None
    delete_old_records = None

    if config:
        chunk_id_prefix = getattr(config, "chunk_id_prefix", None)
        record_retention_hours = getattr(config, "record_retention_hours", None)
        upsert_batch_size = getattr(config, "upsert_batch_size", None)
        delete_old_records = getattr(config, "delete_old_records", None)

    # Initialize the PineconeUploader instance with configuration
    uploader = PineconeUploader(
        api_key,
        index_name,
        chunk_id_prefix=chunk_id_prefix,
        record_retention_hours=record_retention_hours,
        upsert_batch_size=upsert_batch_size,
        delete_old_records=delete_old_records,
    )

    upserted_count = uploader.upsert_records(chunks)
    logger.info(f"Upserted {upserted_count} records")

    deleted_count = uploader.delete_older_than_retention_period()
    logger.info(f"Deleted {deleted_count} old records")

    logger.info(
        f"Operation complete: {upserted_count} records upserted, "
        f"{deleted_count} old records deleted."
    )
