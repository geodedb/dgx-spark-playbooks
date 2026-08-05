FROM ollama/ollama:0.32.6

# Copy the entrypoint script
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Matches the docker-compose healthcheck already defined for this service.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD ollama list || exit 1

