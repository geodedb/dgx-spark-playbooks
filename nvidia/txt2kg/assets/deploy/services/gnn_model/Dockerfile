FROM nvcr.io/nvidia/pyg:26.05-py3

WORKDIR /app

# Install Flask and other required packages
RUN pip install --no-cache-dir \
    flask==2.0.1 \
    gunicorn==23.0.0 \
    tqdm==4.67.3

# Create model directory
RUN mkdir -p /app/models

# Copy application code
COPY services/gnn_model/app.py /app/

# Set environment variables
ENV MODEL_PATH=/app/models/tech-qa-model.pt
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py

# Expose the port
EXPOSE 5000

# app.py exposes GET /health
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=5 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:5000/health', timeout=5).status < 500 else 1)"

# Run the service with gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"] 