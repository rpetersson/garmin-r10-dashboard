# 🐳 Docker Deployment Guide

This document explains how to build and run the Garmin R10 Dashboard using Docker.

## Quick Start

### Using Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/rpetersson/garmin-r10-dashboard.git
cd garmin-r10-dashboard

# Start the application
docker-compose up -d

# Access the dashboard
open http://localhost:8501
```

### Using Docker directly

```bash
# Build the image
docker build -t garmin-r10-dashboard .

# Run the container
docker run -d \
  --name garmin-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  garmin-r10-dashboard

# Access the dashboard
open http://localhost:8501
```

## Using Pre-built Images

Pre-built images are automatically published to GitHub Container Registry:

```bash
# Pull and run the latest image
docker run -d \
  --name garmin-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

## Configuration

### Environment Variables

- `STREAMLIT_SERVER_PORT`: Port for Streamlit (default: 8501)
- `STREAMLIT_SERVER_ADDRESS`: Bind address (default: 0.0.0.0)

### Volume Mounts

- `/app/data`: Mount this directory to persist uploaded CSV files

## Production Deployment

For production deployment, consider:

1. **Reverse Proxy**: Use nginx or traefik for SSL termination
2. **Persistent Storage**: Mount a persistent volume for data
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Health Checks**: The container includes health checks
5. **Logging**: Configure log aggregation

### Example Production Docker Compose

```yaml
version: '3.8'
services:
  garmin-dashboard:
    image: ghcr.io/rpetersson/garmin-r10-dashboard:latest
    restart: unless-stopped
    volumes:
      - /opt/garmin-data:/app/data
    environment:
      - STREAMLIT_SERVER_PORT=8501
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl:ro
    depends_on:
      - garmin-dashboard
    restart: unless-stopped
```

## Development

### Building Locally

```bash
# Build the image
docker build -t garmin-r10-dashboard:dev .

# Run with development settings
docker run -it --rm \
  -p 8501:8501 \
  -v $(pwd):/app \
  garmin-r10-dashboard:dev
```

### Multi-platform Builds

The GitHub Actions workflow builds for both AMD64 and ARM64 architectures:

```bash
# Manual multi-platform build
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t garmin-r10-dashboard:multi \
  --push .
```

## Troubleshooting

### Container Health

```bash
# Check container status
docker ps

# View logs
docker logs garmin-dashboard

# Check health
docker inspect garmin-dashboard --format='{{.State.Health}}'
```

### Common Issues

1. **Port already in use**: Change the host port in docker-compose.yml
2. **Permission issues**: Ensure the data directory is writable
3. **Memory issues**: Increase container memory limits for large datasets

## Security Considerations

- Run as non-root user (included in Dockerfile)
- Use specific image tags instead of `latest` in production
- Regularly update base images for security patches
- Consider using distroless or minimal base images
- Implement proper network segmentation
