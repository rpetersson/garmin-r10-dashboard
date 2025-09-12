# Use Python 3.11 slim for smaller size
FROM python:3.11-slim

# Install minimal dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Create non-root user with proper home directory
RUN groupadd -r streamlit && useradd -r -g streamlit -m -d /home/streamlit streamlit \
    && chown -R streamlit:streamlit /home/streamlit

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .

# Install Python packages with size optimizations
RUN pip install --no-cache-dir --no-compile -r requirements.txt \
    && find /usr/local/lib/python3.11/site-packages -name "*.pyc" -delete \
    && find /usr/local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -name "tests" -type d -exec rm -rf {} + \
    && find /usr/local/lib/python3.11/site-packages -name "test" -type d -exec rm -rf {} + \
    && pip cache purge

# Copy application code and config
COPY app.py .
COPY .streamlit/ .streamlit/

# Create data directory and set permissions
RUN mkdir -p /app/data /tmp/streamlit_uploads && \
    chown -R streamlit:streamlit /app /tmp/streamlit_uploads && \
    chmod -R 755 /app && \
    chmod -R 777 /tmp/streamlit_uploads

# Switch to non-root user
USER streamlit

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV HOME=/home/streamlit

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run Streamlit with permissive upload settings
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.maxUploadSize=1024", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--browser.gatherUsageStats=false"]
