import asyncio
import os
import shutil
import json
import spacy
import hashlib
from urllib.parse import urlparse, urlunparse
from dotenv import load_dotenv
from crawl4ai import (
    AsyncWebCrawler,
    CrawlerRunConfig,
    DefaultMarkdownGenerator,
    LLMConfig,
    LLMContentFilter,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from sanitize_filename import sanitize_filename
from clean_markdown import process_markdown_results
from langchain.schema import Document

# Load environment variables from .env file and spaCy model
load_dotenv()
nlp = spacy.load("en_core_web_sm")

def clean_text(text):
    """Clean and normalize text using spaCy."""
    doc = nlp(text)
    return " ".join(token.text for token in doc)

def extract_keywords(text, num_keywords=5):
    """Extract key phrases from text to use in chunk naming, avoiding Markdown syntax."""
    cleaned_text = text
    for char in ['#', '*', '_', '`', '[', ']', '(', ')', '>', '\n', '$']:
        cleaned_text = cleaned_text.replace(char, ' ')
    
    doc = nlp(cleaned_text)
    keywords = [
        chunk.text.lower() for chunk in doc.noun_chunks
        if not any(c in chunk.text for c in ['[', ']', '(', ')', '#', '*', '`', '\n', '$'])
        and len(chunk.text) > 3
    ]
    keywords.extend([
        ent.text.lower() for ent in doc.ents
        if not any(c in ent.text for c in ['[', ']', '(', ')', '#', '*', '`', '\n', '$'])
        and len(ent.text) > 3
    ])
    
    keyword_freq = {}
    for keyword in keywords:
        if len(keyword.split()) > 1:
            keyword_freq[keyword] = keyword_freq.get(keyword, 0) + 1
    
    top_keywords = sorted(keyword_freq.items(), key=lambda x: x[1], reverse=True)[:num_keywords]
    return [sanitize_filename(k) for k, v in top_keywords] if top_keywords else ["general_content"]

def chunk_documents(docs, chunk_size=700, overlap_ratio=0.3):
    """
    Splits each Document's page_content into fixed-size token chunks using spaCy sentence segmentation.
    Applies an overlap between consecutive chunks for context preservation.
    Adds meaningful chunk names based on content analysis.
    """
    all_chunks = []
    for doc in docs:
        cleaned_content = clean_text(doc.page_content)
        spacy_doc = nlp(cleaned_content)
        sentences = [sent.text.strip() for sent in spacy_doc.sents if sent.text.strip()]

        chunks = []
        chunk_texts = []
        current_chunk = []
        current_chunk_text = []
        current_tokens = 0
        overlap_size = int(chunk_size * overlap_ratio)

        for sent in sentences:
            sent_doc = nlp(sent)
            sent_tokens = [token.text for token in sent_doc]
            sent_len = len(sent_tokens)

            if current_tokens + sent_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                chunk_texts.append(" ".join(current_chunk_text))
                # Create overlap with previous chunk
                current_chunk = current_chunk[-overlap_size:] if overlap_size < len(current_chunk) else current_chunk
                current_chunk_text = current_chunk_text[-overlap_size:] if overlap_size < len(current_chunk_text) else current_chunk_text
                current_tokens = len(current_chunk)

            current_chunk.extend(sent_tokens)
            current_chunk_text.append(sent)
            current_tokens += sent_len

        if current_chunk:
            chunks.append(" ".join(current_chunk))
            chunk_texts.append(" ".join(current_chunk_text))

        for i, (chunk, chunk_text) in enumerate(zip(chunks, chunk_texts)):
            keywords = extract_keywords(chunk_text)
            chunk_name = f"{i+1}_{'_'.join(keywords[:2])}" if keywords else f"{i+1}_chunk"
            metadata = doc.metadata.copy() if hasattr(doc, "metadata") and doc.metadata else {}
            metadata.update({
                "chunk_index": i+1,
                "chunk_name": chunk_name,
                "total_chunks": len(chunks),
                "keywords": keywords,
                "token_count": len(nlp(chunk)),
                "original_url": metadata.get("url", "unknown"),
                "page_title": metadata.get("title", "unknown"),
            })
            chunk_doc = Document(
                page_content=chunk,
                metadata=metadata,
            )
            all_chunks.append(chunk_doc)
    
    return all_chunks

def canonicalize_url(url):
    """Normalize a URL to avoid trivial duplicates (e.g., trailing slashes)."""
    parsed = urlparse(url)
    path = parsed.path.rstrip('/')
    return urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))

def get_content_hash(content):
    """Generate a SHA256 hash for deduplication of content."""
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

# Create a custom deep crawl strategy that implements early URL deduplication
class DeduplicatingBFSCrawlStrategy(BFSDeepCrawlStrategy):
    def __init__(self, max_depth=2, include_external=False, url_filter=None):
        # Only pass the parameters that BFSDeepCrawlStrategy accepts
        super().__init__(max_depth=max_depth, include_external=include_external)
        # Store url_filter separately
        self.url_filter = url_filter
        # Use sets for O(1) lookup time
        self.visited_urls = set()
        self.canonical_urls = set()
    
    async def get_next_urls(self, html, url, depth, **kwargs):
        # First get candidate URLs using the parent method
        next_urls = await super().get_next_urls(html, url, depth, **kwargs)
        
        # Filter out URLs that have been visited
        filtered_urls = []
        for next_url in next_urls:
            canonical = canonicalize_url(next_url)
            if canonical not in self.canonical_urls:
                self.canonical_urls.add(canonical)
                filtered_urls.append(next_url)
                
        return filtered_urls
    
    def mark_url_visited(self, url):
        # Track both original and canonical form
        self.visited_urls.add(url)
        self.canonical_urls.add(canonicalize_url(url))

async def crawl_single_url(url, crawler, config, global_canonical_urls):
    """Process a single URL with duplication checking."""
    canonical_url = canonicalize_url(url)
    
    # Skip if we've already processed this canonical URL
    if canonical_url in global_canonical_urls:
        print(f"Skipping already processed URL: {url}")
        return []
    
    # Mark this URL as processed
    global_canonical_urls.add(canonical_url)
    
    # Run the crawler for this URL
    print(f"Crawling URL: {url}")
    results = await crawler.arun(url, config=config)
    return results

async def main():
    # Check if API key is set in environment
    if "OPENAI_API_KEY" not in os.environ:
        print("⚠️ Warning: OPENAI_API_KEY environment variable is not set.")
        print("Please set your OpenAI API key using:")
        print("    export OPENAI_API_KEY='your-api-key'")
        return

    # Configuration for OpenAI model
    openai_config = LLMConfig(
        provider="openai/gpt-4o-mini",
        api_token=os.environ.get("OPENAI_API_KEY"),
    )

    # Content filter with improved instructions
    content_filter = LLMContentFilter(
        llm_config=openai_config,
        instruction="""
        You are an assistant who is an expert at filtering content extracted from winery 
        websites. You are given a page from a winery website.
        Your task is to extract ONLY substantive content that provides real 
        value to customers visiting the winery website. The purpose of the 
        content is to help customers learn about the winery and its products 
        by using it as RAG for a chatbot.
        
        Include:
        - Any details relevant to a customer visiting the winery website.
        - This could include product details, events, prices, full list of amenities, and other details about the winery.
        
        Exclude:
        - Repeated links such as those in headers and nav sections.
        - "Skip to content" links or accessibility controls.
        - Social media links and sharing buttons.
        - Login/signup sections.
        - Shopping cart elements.
        - Generic welcome messages with no specific information.
        - Breadcrumbs and pagination elements.
        - Header and footer sections that do not contain substantive information.

        FORMAT REQUIREMENTS:
        - Use clear, hierarchical headers (H1, H2, H3).
        - Create concise, scannable bulleted lists for important details.
        - Organize content logically by topic.
        - Preserve exact pricing, dates, hours, and contact information.
        - Remove all navigation indicators like "»" or ">".
        
        Remember: Quality over quantity. Only extract truly useful customer 
        information that directly helps answer questions about visiting, 
        purchasing, or learning about the winery and its products.
        """,
        verbose=True,
    )

    md_generator = DefaultMarkdownGenerator(content_filter=content_filter)

    # Create directories for output
    base_output_dir = "winery_content"
    chunks_output_dir = f"{base_output_dir}/chunks"

    if os.path.exists(base_output_dir):
        print(f"Removing existing '{base_output_dir}' directory...")
        shutil.rmtree(base_output_dir)

    print(f"Creating new output directories...")
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(chunks_output_dir, exist_ok=True)

    # Add this at the top of your main() function
    global_canonical_urls = set()
    global_content_hashes = set()

    async with AsyncWebCrawler() as crawler:
        try:
            # Define starting URLs
            starting_urls = [
                "https://www.westhillsvineyards.com",
                "https://www.westhillsvineyards.com/weddings",
                "https://www.westhillsvineyards.com/wines",
                "https://www.westhillsvineyards.com/tours", 
                "https://www.westhillsvineyards.com/events-and-venues",
                "https://www.westhillsvineyards.com/visit",
                "https://www.westhillsvineyards.com/join"
            ]
            
            # Create a single shared strategy instance 
            shared_strategy = DeduplicatingBFSCrawlStrategy(max_depth=2, include_external=False)
            
            # Create configs that all use the same strategy instance
            configs = []
            for _ in starting_urls:
                config_copy = CrawlerRunConfig(
                    deep_crawl_strategy=shared_strategy,  # Same instance for all
                    scraping_strategy=LXMLWebScrapingStrategy(),
                    markdown_generator=md_generator,
                    excluded_tags=["footer", "nav"],
                    exclude_external_links=True,
                    exclude_social_media_links=True,
                    exclude_external_images=True,
                    verbose=True,
                    js_code=[
                        """
                        (async () => {
                            function isElementHidden(el) {
                                const style = window.getComputedStyle(el);
                                if (style.display === 'none' || style.visibility === 'hidden') {
                                    return true;
                                }
                                if (el.getAttribute('hidden') !== null || el.getAttribute('aria-hidden') === 'true') {
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
                    ]
                )
                configs.append(config_copy)
            
            # Run the crawler with shared strategy
            print(f"Starting parallel crawling of {len(starting_urls)} URLs...")
            tasks = [
                crawl_single_url(url, crawler, cfg, global_canonical_urls) 
                for url, cfg in zip(starting_urls, configs)
            ]
            results_by_url = await asyncio.gather(*tasks)
            
            # Flatten the results
            all_results = [result for sublist in results_by_url for result in sublist]
            
        except Exception as e:
            print(f"❌ Crawling failed due to exception: {e}")
            return

        # Filter for valid pages
        valid_pages = [result for result in all_results if result.status_code == 200]

        # Post-process to remove redundant links
        print("Post-processing results to remove redundant links...")
        processed_pages = process_markdown_results(valid_pages)

        # Step 1: Save the original filtered content
        print("\n📄 Saving original filtered content...")
        saved_count = 0
        langchain_docs = []
        
        for result in processed_pages:
            try:
                # Skip empty markdown files
                if not result.markdown or result.markdown.isspace():
                    print(f"Skipping empty content for {result.url}")
                    continue
                
                # Skip if content hash already exists (duplicate content)
                content_hash = get_content_hash(result.markdown)
                if content_hash in global_content_hashes:
                    print(f"Duplicate content detected for {result.url}")
                    continue
                global_content_hashes.add(content_hash)
                    
                # Generate filename based on URL path
                page_path = sanitize_filename(result.url)
                filename = f"{base_output_dir}/{page_path}.md"

                with open(filename, "w", encoding="utf-8") as f:
                    f.write(result.markdown)
                
                # Extract page title from content if available
                page_title = "Unknown"
                for line in result.markdown.splitlines():
                    if line.startswith("# "):
                        page_title = line[2:].strip()
                        break
                
                # Create Document object for chunking
                doc = Document(
                    page_content=result.markdown,
                    metadata={
                        "url": result.url,
                        "title": page_title,
                        "source_file": filename
                    }
                )
                langchain_docs.append(doc)
                
                print(f"Saved: {filename} (URL: {result.url})")
                saved_count += 1
            except Exception as e:
                print(f"❌ Failed to save page {result.url} due to: {e}")
                continue

        # Step 2: Apply chunking to the documents
        print("\n🔪 Chunking documents...")
        chunks = chunk_documents(langchain_docs, chunk_size=700, overlap_ratio=0.3)
        print(f"Created {len(chunks)} chunks from {len(langchain_docs)} documents")
        
        # Step 3: Save the chunks with metadata-based names
        print("\n💾 Saving chunks with metadata-based names...")
        chunk_index = {}
        
        for i, chunk in enumerate(chunks):
            try:
                metadata = chunk.metadata
                url = metadata.get("original_url", "unknown")
                url_part = url.split("/")[-1] if "/" in url else url
                url_part = sanitize_filename(url_part)[:50]
                keywords = metadata.get("keywords", [])
                sanitized_keywords = [sanitize_filename(k) for k in keywords]
                keyword_part = "_".join(sanitized_keywords[:2]) if sanitized_keywords else f"chunk_{i+1}"
                keyword_part = keyword_part[:50]
                base_filename = f"{chunks_output_dir}/{url_part}_{i+1}"
                if keyword_part:
                    base_filename += f"_{keyword_part}"
                base_filename = sanitize_filename(base_filename)
                filename = f"{base_filename}.md"
                
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(chunk.page_content)
                
                with open(f"{base_filename}_metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                
                chunk_index[filename] = {
                    "metadata": metadata,
                    "token_count": metadata.get("token_count", 0),
                    "keywords": metadata.get("keywords", [])
                }
                
                print(f"Saved chunk: {filename}")
            except Exception as e:
                print(f"Failed with original filename approach: {e}")
                try:
                    source_file = metadata.get("source_file", "unknown")
                    source_name = os.path.basename(source_file).replace(".md", "")
                    fallback_base = f"{chunks_output_dir}/{source_name}_chunk_{i+1}"
                    fallback_filename = f"{fallback_base}.md"
                    
                    with open(fallback_filename, "w", encoding="utf-8") as f:
                        f.write(chunk.page_content)
                    
                    with open(f"{fallback_base}_metadata.json", "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                    
                    chunk_index[fallback_filename] = {
                        "metadata": metadata,
                        "token_count": metadata.get("token_count", 0),
                        "keywords": metadata.get("keywords", [])
                    }
                    
                    print(f"Saved with fallback name: {fallback_filename}")
                    
                except Exception as fallback_error:
                    print(f"Fallback naming also failed: {fallback_error}")
                    try:
                        basic_filename = f"{chunks_output_dir}/chunk_{i+1}.md"
                        with open(basic_filename, "w", encoding="utf-8") as f:
                            f.write(chunk.page_content)
                        
                        with open(f"{chunks_output_dir}/chunk_{i+1}_metadata.json", "w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)
                        
                        chunk_index[basic_filename] = {
                            "metadata": metadata,
                            "token_count": metadata.get("token_count", 0),
                            "keywords": metadata.get("keywords", [])
                        }
                        
                        print(f"Saved with basic fallback name: {basic_filename}")
                    except Exception as e:
                        print(f"❌ All naming attempts failed for chunk {i}. Content could not be saved.")
                        continue
        
        with open(f"{base_output_dir}/chunk_index.json", "w", encoding="utf-8") as f:
            json.dump(chunk_index, f, indent=2)

        print(f"\n✅ Process complete!")
        print(f"- Crawled and processed {saved_count} pages")
        print(f"- Created {len(chunks)} chunks")
        print(f"- Files are saved in the '{base_output_dir}' directory")
        print(f"- Chunks are saved in the '{chunks_output_dir}' directory")
        print(f"- Chunk index is available at '{base_output_dir}/chunk_index.json'")

if __name__ == "__main__":
    asyncio.run(main())