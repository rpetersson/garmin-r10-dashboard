# Quick Start Guide - GitHub Packages

This guide shows you how to quickly run the Garmin R10 Dashboard using the pre-built Docker image from GitHub Container Registry.

## Prerequisites

- Docker installed on your system
- Internet connection to pull the image

## Option 1: Direct Docker Run (Simplest)

```bash
# Create a data directory for your CSV files
mkdir -p data

# Run the container
docker run -d \
  --name garmin-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest

# Access the application at http://localhost:8501
```

## Option 2: Using Docker Compose (Recommended)

1. **Create a docker-compose.yml file:**

```yaml
version: '3.8'

services:
  garmin-dashboard:
    image: ghcr.io/rpetersson/garmin-r10-dashboard:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
      - STREAMLIT_SERVER_ADDRESS=0.0.0.0
    restart: unless-stopped
```

2. **Run the application:**

```bash
# Create data directory
mkdir -p data

# Start the container
docker-compose up -d

# Access the application at http://localhost:8501
```

## Option 3: Using the Provided Files

If you have cloned or downloaded this repository:

```bash
# Navigate to the project directory
cd garmin-r10-dashboard

# Use the provided Makefile
make pull    # Pull the latest image
make run     # Run the container

# Or use docker-compose
make compose
```

## Managing the Container

### View logs:
```bash
docker logs -f garmin-dashboard
```

### Stop the container:
```bash
docker stop garmin-dashboard
docker rm garmin-dashboard
```

### Update to latest version:
```bash
# Stop current container
docker stop garmin-dashboard
docker rm garmin-dashboard

# Pull latest image
docker pull ghcr.io/rpetersson/garmin-r10-dashboard:latest

# Run with updated image
docker run -d --name garmin-dashboard -p 8501:8501 -v $(pwd)/data:/app/data ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

## Using the Application

1. **Open your browser** to http://localhost:8501
2. **Upload CSV files** from your Garmin Approach R10 using the sidebar
3. **Explore the analytics** across the different tabs
4. **Your data persists** in the `./data` directory between container restarts

## Available Image Tags

- `latest` - Latest stable release
- `main` - Latest from main branch (may be unstable)
- `v1.0.0` - Specific version tags

## Troubleshooting

### Port already in use:
```bash
# Use a different port
docker run -d --name garmin-dashboard -p 8502:8501 -v $(pwd)/data:/app/data ghcr.io/rpetersson/garmin-r10-dashboard:latest
# Access at http://localhost:8502
```

### Permission issues with data directory:
```bash
# Make sure the data directory is writable
mkdir -p data
chmod 755 data
```

### Container won't start:
```bash
# Check container logs
docker logs garmin-dashboard

# Check if container is running
docker ps -a
```

## Advanced Usage

### Custom configuration:
```bash
# Run with custom Streamlit config
docker run -d \
  --name garmin-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -e STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200 \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

### Health check:
```bash
# Test if the application is healthy
curl http://localhost:8501/_stcore/health
```

## Next Steps

- Check out the full [README.md](README.md) for detailed features
- Explore the [DOCKER.md](DOCKER.md) for Docker-specific information
- Read [PACKAGE.md](PACKAGE.md) for GitHub Packages details
