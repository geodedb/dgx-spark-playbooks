FROM python:3.15-slim

WORKDIR /app

# Copy requirements and install dependencies first for better caching
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py /app/

# Set default model name
ENV MODEL_NAME="all-MiniLM-L6-v2"
ENV TRANSFORMERS_CACHE="/app/.cache"

# Pre-download the model during build for faster startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${MODEL_NAME}')"

# Expose the port
EXPOSE 80

# Drop root privileges for the running process. Gunicorn binds to port 80
# (a privileged port), so grant the python interpreter CAP_NET_BIND_SERVICE
# via setcap rather than running the service as root.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libcap2-bin && \
    rm -rf /var/lib/apt/lists/* && \
    useradd --create-home --uid 1000 appuser && \
    setcap 'cap_net_bind_service=+ep' "$(readlink -f "$(command -v python3)")" && \
    chown -R appuser:appuser /app
USER appuser

# Matches the docker-compose healthcheck already defined for this service.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:80/health', timeout=5).status < 500 else 1)"

# Use Gunicorn for better performance
CMD ["gunicorn", "--bind", "0.0.0.0:80", "--workers", "1", "--threads", "8", "app:app"] 