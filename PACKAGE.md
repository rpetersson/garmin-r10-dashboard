# 🏌️ Garmin R10 Dashboard - Docker Package

A comprehensive Streamlit dashboard for analyzing Garmin R10 golf launch monitor data.

## 📦 GitHub Package Information

**Registry:** `ghcr.io`  
**Package:** `ghcr.io/rpetersson/garmin-r10-dashboard`  
**Supported Architectures:** `linux/amd64`, `linux/arm64`

## 🚀 Quick Start

### Pull and Run

```bash
# Pull the latest image
docker pull ghcr.io/rpetersson/garmin-r10-dashboard:latest

# Run the dashboard
docker run -d \
  --name garmin-dashboard \
  -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  garmin-dashboard:
    image: ghcr.io/rpetersson/garmin-r10-dashboard:latest
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
```

## 🏷️ Available Tags

- `latest` - Latest stable release
- `main` - Latest development build
- `v1.0.0`, `v1.1.0`, etc. - Specific version releases

## 📊 Features

- **Performance Analysis:** Track trends in distance, accuracy, and consistency
- **Shot Shape Analysis:** Automatic classification of draws, fades, hooks, and slices
- **Iron Attack Angle:** Specialized analysis for optimal iron performance (-3° target)
- **Spin Analysis:** Backspin and sidespin trends with optimal zones
- **Environmental Impact:** Temperature and air density effects
- **Multi-Session Support:** Compare performance across different practice sessions

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMLIT_SERVER_PORT` | `8501` | Port for the Streamlit server |
| `STREAMLIT_SERVER_ADDRESS` | `0.0.0.0` | Bind address for the server |

### Volume Mounts

| Container Path | Description |
|----------------|-------------|
| `/app/data` | Directory for CSV file uploads and persistence |

## 🛡️ Security Features

- Runs as non-root user (`streamlit:streamlit`)
- Multi-stage build for minimal attack surface
- Regular security scanning with Trivy
- Signed container images with build attestation

## 📋 System Requirements

- Docker 20.10+ or Docker Desktop
- 512MB RAM minimum (1GB+ recommended for large datasets)
- Port 8501 available (or configure alternative)

## 🔍 Health Checks

The container includes built-in health checks:

```bash
# Check container health
docker inspect garmin-dashboard --format='{{.State.Health.Status}}'

# View health check logs
docker inspect garmin-dashboard --format='{{range .State.Health.Log}}{{.Output}}{{end}}'
```

## 📝 Usage Examples

### Development Mode

```bash
# Run with live reload for development
docker run -it --rm \
  -p 8501:8501 \
  -v $(pwd):/app \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

### Production Deployment

```bash
# Production deployment with resource limits
docker run -d \
  --name garmin-dashboard-prod \
  --restart unless-stopped \
  --memory=1g \
  --cpus=0.5 \
  -p 8501:8501 \
  -v /opt/garmin-data:/app/data \
  ghcr.io/rpetersson/garmin-r10-dashboard:latest
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: garmin-dashboard
spec:
  replicas: 1
  selector:
    matchLabels:
      app: garmin-dashboard
  template:
    metadata:
      labels:
        app: garmin-dashboard
    spec:
      containers:
      - name: dashboard
        image: ghcr.io/rpetersson/garmin-r10-dashboard:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: garmin-data-pvc
```

## 🐛 Troubleshooting

### Common Issues

1. **Port already in use**
   ```bash
   # Use a different port
   docker run -p 8502:8501 ghcr.io/rpetersson/garmin-r10-dashboard:latest
   ```

2. **Permission denied on data volume**
   ```bash
   # Fix volume permissions
   sudo chown -R 1000:1000 ./data
   ```

3. **Container exits immediately**
   ```bash
   # Check logs
   docker logs garmin-dashboard
   ```

### Performance Tuning

For large datasets (1000+ shots):
- Increase memory limit: `--memory=2g`
- Use SSD storage for data volume
- Consider horizontal scaling for multiple users

## 📚 Documentation

- [Full Documentation](https://github.com/rpetersson/garmin-r10-dashboard)
- [Docker Guide](https://github.com/rpetersson/garmin-r10-dashboard/blob/main/DOCKER.md)
- [API Reference](https://docs.streamlit.io/)

## 🤝 Contributing

Found a bug or want to contribute? Visit the [GitHub repository](https://github.com/rpetersson/garmin-r10-dashboard) to open issues or submit pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/rpetersson/garmin-r10-dashboard/blob/main/LICENSE) file for details.
