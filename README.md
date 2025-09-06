# Web Crawler with Vector Database Integration

This project implements a web crawler that processes content from websites and stores it in a vector database for efficient semantic search capabilities. It's designed for crawling winery websites and organizing their content for RAG (Retrieval-Augmented Generation) applications.

## Features

- Specialized content crawling using crawl4ai
- Intelligent content filtering using OpenAI models
- Advanced web crawling with configurable depth
- Duplicate link detection
- JavaScript execution for handling dynamic content
- Markdown content generation
- Robust error handling and logging
- **Flexible configuration system** with environment variables and YAML support
- **Docker support** for easy deployment in any environment

## Prerequisites

- Python 3.9+ (for local development)
- OpenAI API access (for content filtering)
- Docker and Docker Compose (for containerized deployment) 

## Setup

### Option 1: Local Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/web-crawler.git
cd web-crawler
```

2. Set up a virtual environment:
```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install required packages
pip install -r requirements.txt
```

4. Install the package in development mode to handle absolute imports:
```bash
# This allows the project's absolute imports to work correctly
pip install -e .
```

5. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Adjust other configuration parameters as needed

### Option 2: Docker Setup (Recommended for Production)

1. Clone the repository:
```bash
git clone https://github.com/yourusername/web-crawler.git
cd web-crawler
```

2. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```
   - Adjust other configuration parameters as needed

3. Make the deployment script executable:
```bash
chmod +x deploy.sh
```

4. Run the crawler using the deployment script:
```bash
./deploy.sh
```

This will:
- Create necessary directories
- Build the Docker image with all dependencies (including browsers)
- Run the crawler in a container

You can pass any arguments to the script and they'll be forwarded to the orchestrator:
```bash
./deploy.sh --dry-run
./deploy.sh --config examples/winery_config.yaml
```

## Configuration

The crawler supports a flexible configuration system that can be customized through:

1. **Environment Variables**: Set in `.env` file or system environment
2. **YAML Configuration Files**: Create custom configurations for different use cases
3. **Command-line Arguments**: Override settings when running the crawler

### Configuration Parameters

The configuration system allows you to customize all aspects of the crawler's behavior:

#### Crawler Settings
- `START_URLS`: Comma-separated list of starting URLs
- `MAX_DEPTH`: How deep to crawl from starting URLs (0 means only starting URLs)
- `INCLUDE_EXTERNAL`: Whether to follow links to external domains
- `BATCH_SIZE`: Number of URLs to process in parallel
- `EXCLUDED_TAGS`: HTML tags to exclude from processing

#### Browser Settings
- `USE_BROWSER`: Whether to use a browser (true) or HTTP-only mode (false)
- `BROWSER_TYPE`: Browser engine to use (chromium, firefox, webkit)
- `HEADLESS`: Whether to run the browser in headless mode
- `LIGHT_MODE`: Performance optimization for low-resource environments
- `TEXT_MODE`: Skip loading images to improve performance

#### Content Processing
- `CHUNK_SIZE`: Size of content chunks for vector storage
- `CHUNK_OVERLAP_RATIO`: Overlap between chunks to maintain context
- `SUMMARY_MODEL_NAME`: OpenAI model to use for summarization

#### Vector Database
- `CHUNK_ID_PREFIX`: Prefix for chunk IDs in vector database
- `RECORD_RETENTION_HOURS`: How long to keep old records before deletion
- `UPSERT_BATCH_SIZE`: Number of records to upsert in each batch
- `DELETE_OLD_RECORDS`: Whether to delete outdated records

### Example Configuration Files

The `examples/` directory contains sample YAML configurations for different use cases:

- `winery_config.yaml`: Optimized for crawling winery websites
- `ecommerce_config.yaml`: Configured for e-commerce websites
- `render_config.yaml`: Optimized for deployment on Render (uses HTTP-only mode)

### Running with Custom Configuration

You can run the orchestrator with a custom YAML configuration:

```bash
# With Docker:
./deploy.sh --config examples/winery_config.yaml

# Without Docker:
python orchestrator.py --config examples/winery_config.yaml
```

Or use environment variables with the default settings:

```bash
# Set environment variables directly
export START_URLS="https://example.com/page1,https://example.com/page2"
export MAX_DEPTH=2
python orchestrator.py
```

### Running in Dry-Run Mode

To save results locally without uploading to Pinecone:

```bash
# With Docker:
./deploy.sh --dry-run

# Without Docker:
python orchestrator.py --dry-run
```

## Docker Customization

The Docker setup provides several ways to customize the crawler:

### Environment Variables

You can override any environment variable in the `docker-compose.yml` file:

```yaml
environment:
  - OPENAI_API_KEY=your_api_key_here
  - START_URLS=https://example.com/page1,https://example.com/page2
  - MAX_DEPTH=2
  - USE_BROWSER=true
```

### Mounted Volumes

The Docker Compose configuration mounts the following directories:

- `./cleaned_output:/app/cleaned_output`: Stores processed content
- `./logs:/app/logs`: Stores log files

These directories will persist between container runs.

### Custom Commands

You can run custom commands with Docker Compose:

```bash
# Run with a specific config file
docker-compose run --rm crawler python orchestrator.py --config examples/winery_config.yaml

# Run in dry-run mode
docker-compose run --rm crawler python orchestrator.py --dry-run
```

## Orchestration Process

The orchestrator runs the complete pipeline:

1. **Crawling**: Fetches content from specified URLs
2. **Chunking**: Splits content into manageable chunks
3. **Summarization**: Processes chunks and extracts keywords
4. **Vector DB Upload**: Uploads processed content to Pinecone (or saves locally in dry-run mode)

## Usage

To run the complete process:

```bash
# Run with default settings
python orchestrator.py

# Run with custom configuration
python orchestrator.py --config examples/winery_config.yaml

# Run in dry-run mode (save locally)
python orchestrator.py --dry-run
```

## Production Deployment

For production environments, follow these best practices:

### Installation

Instead of development mode installation, install the package properly:

1. Create a `setup.py` file in the root directory if it doesn't exist:
```python
from setuptools import setup, find_packages

setup(
    name="web-crawler",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "crawl4ai",
        "openai",
        "python-dotenv",
        # Add other dependencies from requirements.txt
    ],
)
```

2. Build and install the package:
```bash
pip install .
```

### Environment Configuration

There are several approaches to managing environment variables in production, each with its own advantages:

#### Option 1: Using `.env` files (Simple Setup)

`.env` files can be used in production with proper security measures:

```bash
# Create a .env file with restricted permissions
touch .env
chmod 600 .env  # Only the owner can read/write

# Add your environment variables
echo "OPENAI_API_KEY=your_api_key_here" >> .env
```

**Security considerations:**
- Store `.env` files outside of version control
- Use restricted file permissions
- Consider encrypting the file when not in use
- Establish a secure process for deploying `.env` files

#### Option 2: System Environment Variables (Recommended for many deployments)

Use system-level environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
# Add any other required environment variables
```

Add to startup scripts for persistence:
```bash
# Add to ~/.bashrc or ~/.bash_profile for persistence
echo 'export OPENAI_API_KEY=your_api_key_here' >> ~/.bashrc
```

#### Option 3: Container Environment Variables

For containerized deployments (e.g., Docker):

```bash
# When running the container
docker run -e OPENAI_API_KEY=your_api_key_here web-crawler

# Or using an env file with Docker
docker run --env-file .env web-crawler
```

#### Option 4: Secrets Management Systems (Enterprise)

For enterprise environments, use a dedicated secrets management system:
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault
- Google Secret Manager

These provide enhanced security features like:
- Encryption at rest and in transit
- Access controls and auditing
- Automatic rotation
- Integration with identity management

Choose the approach that best fits your security requirements and operational complexity.

### Containerization

Using Docker for containerization:

1. Create a `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir .

CMD ["python", "-m", "crawler.crawl"]
```

2. Build and run the container:
```bash
docker build -t web-crawler .
docker run -e OPENAI_API_KEY=your_api_key_here web-crawler
```

### Error Handling and Logging

For production, enhance the error handling and logging:

1. Add structured logging to a file:
```python
import logging

logging.basicConfig(
    filename='crawler.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

2. Implement proper exception handling with retry mechanisms for transient failures.

### Performance Optimization

1. Adjust batch size and concurrency based on your infrastructure:
   - For higher-end servers, increase batch size
   - For limited resources, decrease concurrency

2. Consider implementing rate limiting to respect website policies:
```python
# Add delay between requests
config = CrawlerRunConfig(
    # existing config...
    request_delay=1.0,  # 1 second delay between requests
)
```

### Monitoring

1. Implement health checks to monitor crawler performance
2. Add metrics collection for:
   - Pages crawled
   - Success/failure rates
   - Processing time
   - API usage (for OpenAI)

3. Setup alerts for critical failures

### Scheduling

For periodic crawling, use a scheduling system:

1. Simple scheduling with cron:
```bash
# Run crawler every day at 2 AM
0 2 * * * cd /path/to/web-crawler && python -m crawler.crawl
```

2. For more complex workflows, consider Airflow or similar orchestration tools.

## Project Structure

```
web-crawler/
├── crawler/
│   ├── __init__.py
│   ├── crawl.py          # Main crawler implementation
│   ├── sanitize_filename.py
│   └── clean_markdown.py
├── setup.py              # Package installation configuration
├── requirements.txt      # Project dependencies
└── .env                  # Environment variables (create from .env.example)
```

## Environmental Variables

Required environment variables:
- `OPENAI_API_KEY`: Your OpenAI API key for content filtering

## Advanced Configuration

You can modify the crawler behavior by editing the configuration in `crawler/crawl.py`:

- Change the starting URL (currently set to "https://www.westhillsvineyards.com/wines")
- Adjust crawl depth for more comprehensive content collection
- Modify batch size for performance tuning
- Update content filtering instructions
- Add specific tag exclusions
- Configure JavaScript code for DOM manipulation before crawling

## Troubleshooting

If you encounter import errors when running the crawler:

1. Ensure you've installed the package in development mode:
   ```bash
   pip install -e .
   ```

2. Verify that your virtual environment is activated:
   ```bash
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. Check that the project structure follows the expected pattern with proper `__init__.py` files in each directory.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

