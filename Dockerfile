FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY Anti-Scam-Agent-main/Anti-Scam-Agent-main/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code from the nested directory structure
COPY Anti-Scam-Agent-main/Anti-Scam-Agent-main/ .

# Create logs directory
RUN mkdir -p logs

# Default port (Render will override this via environment)
ENV PORT=8000

# Expose port
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Run the application - PORT is set by Render automatically
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT}
