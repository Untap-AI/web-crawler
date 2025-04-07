#!/usr/bin/env python3
"""
Standalone Pinecone Upload Script

This script takes markdown files generated by a web scraper and uploads them to Pinecone.
It's designed to be run as a separate step after the web scraping process and works with
the enhanced chunking functionality that generates metadata JSON files.

Usage:
    python -m vectordb.upload_to_pinecone [--dry] [--input-dir DIRECTORY]

Options:
    --dry           Run in dry run mode (process but don't upload to Pinecone)
    --input-dir     Directory containing markdown files (default: winery_content)
"""

import os
import sys
import glob
import json
import argparse
import logging
import uuid
import time
import shutil
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re
from dotenv import load_dotenv
import pinecone  # Updated import for the new package name

# Add parent directory to path to allow importing from crawler package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class PineconeUploader:
    """Handles uploading content to Pinecone."""

    def __init__(
        self,
        api_key: str,
        environment: str = "us-east-1",
        index_name: str = None,
        namespace: str = "default",
        embedding_model: str = "llama-text-embed-v2",
    ):
        """
        Initialize the PineconeUploader.

        Args:
            api_key: Pinecone API key
            environment: Pinecone environment (e.g., us-east-1)
            index_name: Name of the Pinecone index
            namespace: Namespace for the vectors
            embedding_model: Name of the embedding model to use for embedding chunks
        """
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.namespace = namespace
        self.embedding_model = embedding_model

        # Initialize Pinecone client
        self.pc = pinecone.Pinecone(api_key=self.api_key)

        # Verify index exists
        if not self.pc.has_index(self.index_name):
            raise ValueError(f"Index '{self.index_name}' does not exist in Pinecone")
        logger.info(f"Using existing index: {self.index_name}")

    def upsert_records(self, records: List[Dict[str, Any]]) -> int:
        """
        Upsert records to the Pinecone index.

        Args:
            records: List of records to upsert

        Returns:
            Number of records upserted
        """
        # Get the index
        index = self.pc.index(self.index_name)

        # Upsert records with embedding
        logger.info(f"Upserting {len(records)} records to namespace '{self.namespace}'")
        index.upsert_records(
            records=records,
            namespace=self.namespace,
            embed={
                "model": self.embedding_model,
                "field_map": {"text": "chunk_text"},
            },
        )

        # Wait a moment for consistency
        time.sleep(2)

        # Get index stats
        stats = index.describe_index_stats()
        logger.info(f"Index stats after upsert: {stats}")

        return len(records)

    def delete_old_records(
        self,
        customer_id: str,
        source_type: str = "web_crawl",
        exclude_batch_id: str = None,
    ) -> int:
        """
        Delete only existing records for a specific customer that match the source type.

        Args:
            customer_id: Customer identifier used to filter records to delete
            source_type: Source type identifier (default: "web_crawl") to ensure only
                         crawled website content is deleted, preserving other data
            exclude_batch_id: Optional batch ID to exclude from deletion

        Returns:
            Number of records deleted
        """
        # Get the index
        index = self.pc.index(self.index_name)

        # Get current count before deletion for reporting
        before_stats = index.describe_index_stats()
        before_count = (
            before_stats.get("namespaces", {})
            .get(self.namespace, {})
            .get("vector_count", 0)
        )

        # Create filter for customer_id AND source_type to precisely target only web crawl data
        if exclude_batch_id:
            # If a batch ID is provided, exclude it from deletion
            filter_dict = {
                "$and": [
                    {"customer_id": {"$eq": customer_id}},
                    {"source_type": {"$eq": source_type}},
                    {"batch_id": {"$ne": exclude_batch_id}},
                ]
            }
        else:
            # Delete all records of this type
            filter_dict = {
                "$and": [
                    {"customer_id": {"$eq": customer_id}},
                    {"source_type": {"$eq": source_type}},
                ]
            }

        # Delete records matching filter
        logger.info(
            f"Deleting old {source_type} records for customer '{customer_id}' from namespace '{self.namespace}'"
        )
        index.delete(filter=filter_dict, namespace=self.namespace)

        # Wait a moment for consistency
        time.sleep(2)

        # Get after count
        after_stats = index.describe_index_stats()
        after_count = (
            after_stats.get("namespaces", {})
            .get(self.namespace, {})
            .get("vector_count", 0)
        )

        # Calculate deleted count
        deleted_count = before_count - after_count

        logger.info(
            f"Deleted {deleted_count} old {source_type} records for customer '{customer_id}'"
        )
        logger.info(f"Index stats after deletion: {after_stats}")

        return deleted_count

    def replace_records(
        self,
        customer_id: str,
        records: List[Dict[str, Any]],
        source_type: str = "web_crawl",
    ) -> Tuple[int, int]:
        """
        Replace only web crawl records for a customer by first upserting new records
        then deleting old ones with the same source type.

        Args:
            customer_id: Customer identifier
            records: New records to upsert
            source_type: Source type identifier to ensure only specific content is replaced

        Returns:
            Tuple containing (number of records upserted, number of records deleted)
        """
        # Generate a batch_id to identify this specific upload batch
        batch_id = f"batch_{uuid.uuid4().hex}"
        current_time = datetime.now().isoformat()

        # Add batch_id and upload_timestamp to all records
        for record in records:
            record["metadata"]["batch_id"] = batch_id
            record["metadata"]["upload_timestamp"] = current_time

        # First upsert new records
        upserted_count = self.upsert_records(records)

        # Now delete old records for this customer, EXCLUDING this batch
        filter_dict = {
            "$and": [
                {"customer_id": {"$eq": customer_id}},
                {"source_type": {"$eq": source_type}},
                {
                    "batch_id": {"$ne": batch_id}
                },  # Only select records NOT in this batch
            ]
        }

        # Get the index
        index = self.pc.index(self.index_name)

        # Get current count before deletion for reporting
        before_stats = index.describe_index_stats()
        before_count = (
            before_stats.get("namespaces", {})
            .get(self.namespace, {})
            .get("vector_count", 0)
        )

        # Delete the old records that don't have this batch ID
        logger.info(
            f"Deleting old {source_type} records for customer '{customer_id}' from namespace '{self.namespace}'"
        )
        index.delete(filter=filter_dict, namespace=self.namespace)

        # Wait a moment for consistency
        time.sleep(2)

        # Get after count
        after_stats = index.describe_index_stats()
        after_count = (
            after_stats.get("namespaces", {})
            .get(self.namespace, {})
            .get("vector_count", 0)
        )

        # Calculate deleted count
        deleted_count = before_count - after_count

        logger.info(
            f"Replaced {source_type} records for customer '{customer_id}': {upserted_count} added, {deleted_count} old ones removed"
        )

        return upserted_count, deleted_count


def read_markdown_file(file_path: str) -> str:
    """
    Read content from a markdown file.

    Args:
        file_path: Path to the markdown file

    Returns:
        Content of the markdown file
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_json_metadata(file_path: str) -> Dict[str, Any]:
    """
    Load metadata from a JSON file.

    Args:
        file_path: Path to the JSON metadata file

    Returns:
        Dictionary containing metadata
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load metadata from {file_path}: {e}")
        return {}


def get_metadata_file_path(markdown_file_path: str) -> str:
    """
    Generate the expected path to the metadata JSON file for a given markdown file.

    Args:
        markdown_file_path: Path to the markdown file

    Returns:
        Path to the expected metadata JSON file
    """
    base_path = os.path.splitext(markdown_file_path)[0]
    return f"{base_path}_metadata.json"


def check_for_chunk_index(input_dir: str) -> Optional[Dict[str, Any]]:
    """
    Check for and load the chunk index file.

    Args:
        input_dir: Directory containing the chunk index

    Returns:
        Dictionary containing the chunk index if found, None otherwise
    """
    index_path = os.path.join(input_dir, "chunk_index.json")
    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load chunk index from {index_path}: {e}")
    return None


def extract_metadata_from_file(
    file_path: str, customer_id: str
) -> Tuple[Dict[str, Any], str]:
    """
    Extract metadata and content from a markdown file.

    Args:
        file_path: Path to the markdown file
        customer_id: Customer identifier

    Returns:
        Tuple containing (metadata dictionary, content string)
    """
    # Read the markdown file
    content = read_markdown_file(file_path)

    # Check if we have a companion metadata file
    metadata_file_path = file_path
    if file_path.endswith(".md"):
        metadata_file_path = file_path[:-3] + "_metadata.json"

    url = None
    title = None
    chunk_id = None

    # First try to load metadata from the companion JSON file
    if os.path.exists(metadata_file_path):
        try:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                metadata_json = json.load(f)
                url = metadata_json.get("url")
                title = metadata_json.get("title") or metadata_json.get("page_title")
                chunk_id = metadata_json.get("chunk_id")
                logger.info(
                    f"Loaded metadata from companion file: {metadata_file_path}"
                )
        except Exception as e:
            logger.warning(f"Error loading metadata from {metadata_file_path}: {e}")

    # If no metadata file or missing fields, try to extract from content
    if not url:
        url_match = re.search(r"^URL: (https?://[^\s]+)", content, re.MULTILINE)
        url = url_match.group(1) if url_match else None

    if not title:
        title_match = re.search(r"^Title: (.+)", content, re.MULTILINE)
        if title_match:
            title = title_match.group(1)
        else:
            # Try to extract first heading as title
            heading_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
            title = heading_match.group(1) if heading_match else None

    # Extract chunk_id from filename if not found in metadata
    if not chunk_id:
        if "_chunk_" in file_path:
            chunk_match = re.search(r"_chunk_(\d+)", file_path)
            if chunk_match:
                chunk_id = chunk_match.group(1)
        else:
            # If not a chunk file, use the filename as chunk_id
            chunk_id = os.path.splitext(os.path.basename(file_path))[0]

    # Create metadata dictionary
    metadata = {
        "customer_id": customer_id,
        "url": url,
        "title": title,
        "chunk_id": chunk_id,
        "source_type": "web_crawl",
    }

    # Remove metadata lines from content if they exist
    content = re.sub(r"^URL: .+\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"^Title: .+\n", "", content, flags=re.MULTILINE)

    return metadata, content


def save_chunks_to_disk(chunks: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Save processed chunks to disk in dry run mode.

    Args:
        chunks: List of processed chunks
        output_dir: Directory to save processed content
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Clean up any existing processed content files - use *.json to catch ALL existing files
    existing_files = glob.glob(os.path.join(output_dir, "*.json"))
    if existing_files:
        logger.info(f"Found {len(existing_files)} existing files to clean up")
        for file_path in existing_files:
            try:
                os.remove(file_path)
                logger.info(f"Removed old processed content file: {file_path}")
            except Exception as e:
                logger.error(f"Error removing old file {file_path}: {str(e)}")

    # Save new chunks
    output_file = os.path.join(
        output_dir, f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(chunks)} processed chunks to {output_file}")
    except Exception as e:
        logger.error(f"Error saving processed chunks: {str(e)}")


def process_markdown_files(input_dir: str, dry_run: bool = False) -> int:
    """
    Process markdown files from a directory and optionally upload them to Pinecone.

    Args:
        input_dir: Directory containing markdown files to process
        dry_run: If True, process but don't upload to Pinecone

    Returns:
        Number of processed chunks
    """
    # Load environment variables
    load_dotenv()

    # Check required environment variables
    required_vars = ["PINECONE_API_KEY", "CUSTOMER_ID"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        logger.error(
            f"Missing required environment variables: {', '.join(missing_vars)}"
        )
        return 0

    # Get environment variables
    api_key = os.getenv("PINECONE_API_KEY")
    customer_id = os.getenv("CUSTOMER_ID")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    index_name = os.getenv("PINECONE_INDEX_NAME", f"rag-{customer_id.lower()}")
    namespace = os.getenv("NAMESPACE", "default")
    embedding_model = os.getenv("EMBEDDING_MODEL", "llama-text-embed-v2")

    # Check if directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Directory not found: {input_dir}")
        return 0

    # Convert the relative input_dir to an absolute path
    if not os.path.isabs(input_dir):
        # If it's a relative path with "crawler/" prefix, use it as is
        if not input_dir.startswith("crawler/"):
            input_dir = os.path.join("crawler", input_dir)

    chunks_dir = os.path.join(input_dir, "chunks")

    # Set the output directory for dry run mode
    output_dir = os.getenv("OUTPUT_DIR", "vectordb/processed_content")

    # Check for chunk index file
    chunk_index = check_for_chunk_index(input_dir)

    # Check for chunks directory
    use_pre_chunked = os.path.exists(chunks_dir)

    if use_pre_chunked:
        logger.info(f"Found pre-chunked content in {chunks_dir}")

        if chunk_index:
            logger.info(f"Found chunk index with {len(chunk_index)} entries")

        # Get all chunk files
        chunk_files = glob.glob(os.path.join(chunks_dir, "*.md"))
        if not chunk_files:
            logger.warning(f"No chunk files found in {chunks_dir}")
            use_pre_chunked = False

    # If no pre-chunked content, use regular markdown files
    if not use_pre_chunked:
        logger.info(
            f"No pre-chunked content found, using regular markdown files from {input_dir}"
        )
        markdown_files = glob.glob(os.path.join(input_dir, "*.md"))

        if not markdown_files:
            logger.warning(f"No markdown files found in {input_dir}")
            return 0

        logger.info(f"Found {len(markdown_files)} markdown files in {input_dir}")
    else:
        logger.info(f"Found {len(chunk_files)} pre-chunked files in {chunks_dir}")

    # Process files
    all_chunks = []

    if use_pre_chunked:
        # Process pre-chunked files
        for file_path in chunk_files:
            try:
                # Extract metadata and content using companion JSON if available
                metadata, content = extract_metadata_from_file(file_path, customer_id)

                # Create chunk record
                chunk = {"chunk_text": content, "metadata": metadata}
                all_chunks.append(chunk)

                logger.info(f"Processed pre-chunked file: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue
    else:
        # Process regular markdown files (no chunking - each file becomes one document)
        for file_path in markdown_files:
            try:
                # Extract metadata and content
                metadata, content = extract_metadata_from_file(file_path, customer_id)

                # Create single chunk for the entire file
                chunk = {"chunk_text": content, "metadata": metadata}
                all_chunks.append(chunk)

                logger.info(f"Processed full document: {file_path}")
            except Exception as e:
                logger.error(f"Error processing {file_path}: {str(e)}")
                continue

    # Handle chunks based on mode
    if dry_run:
        # Generate batch_id for dry run
        batch_id = f"batch_{uuid.uuid4().hex}"
        current_time = datetime.now().isoformat()

        # Add batch_id and timestamp to all chunks
        for chunk in all_chunks:
            chunk["metadata"]["batch_id"] = batch_id
            chunk["metadata"]["upload_timestamp"] = current_time
            chunk["metadata"]["source_type"] = "web_crawl"

        # Dry run mode: save to disk
        save_chunks_to_disk(all_chunks, output_dir)
    else:
        # Embedding mode: upload to Pinecone
        try:
            # Initialize Pinecone uploader
            uploader = PineconeUploader(
                api_key=api_key,
                environment=environment,
                index_name=index_name,
                namespace=namespace,
                embedding_model=embedding_model,
            )

            # Convert chunks to Pinecone records format
            records = []
            for chunk in all_chunks:
                record = {
                    "id": chunk["metadata"].get("chunk_id", str(uuid.uuid4())),
                    "chunk_text": chunk["chunk_text"],
                    "metadata": {},
                }
                # Add all metadata as fields
                for key, value in chunk["metadata"].items():
                    # Skip any overly complex metadata that can't be stored in Pinecone
                    if (
                        isinstance(value, (str, int, float, bool, list))
                        or value is None
                    ):
                        record["metadata"][key] = value

                records.append(record)

            # Replace existing records with new ones
            uploader.replace_records(customer_id, records)
        except Exception as e:
            logger.error(f"Error uploading to Pinecone: {str(e)}")
            return 0

    return len(all_chunks)


def main():
    """Main entry point for the script."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Upload markdown files to Pinecone")
    parser.add_argument(
        "--dry", action="store_true", help="Run in dry mode (process but don't upload)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="winery_content",
        help="Directory containing markdown files",
    )
    args = parser.parse_args()

    try:
        # Process markdown files
        num_processed = process_markdown_files(
            input_dir=args.input_dir, dry_run=args.dry
        )

        if num_processed > 0:
            logger.info(f"Successfully processed {num_processed} chunks")
            return 0
        else:
            logger.error("No files were processed")
            return 1
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
