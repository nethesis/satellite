# Build stage
FROM docker.io/library/python:3.14.4-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Copy application files
COPY *.py /tmp/
COPY agent/ /tmp/agent/
COPY README.md /tmp/

# Install dependencies
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt

# Final stage
FROM docker.io/library/python:3.14.4-slim

# Install runtime dependencies for PyAudio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libportaudio2 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application files
COPY --from=builder /tmp/*.py /app/
COPY --from=builder /tmp/agent/ /app/agent/
COPY --from=builder /tmp/README.md /app/

# Make sure scripts in .local are usable
ENV PATH=/opt/venv/bin:$PATH

# Set environment variables with default values (can be overridden at runtime)
ENV ASTERISK_URL="http://127.0.0.1:8088" \
    ARI_APP="satellite" \
    ARI_USERNAME="satellite" \
    SATELLITE_ARI_PASSWORD="dummypassword" \
    ASTERISK_FORMAT="slin16" \
    RTP_HOST="127.0.0.1" \
    RTP_PORT="10000" \
    RTP_SWAP16="true" \
    RTP_HEADER_SIZE="12" \
    MQTT_URL="mqtt://127.0.0.1:1883" \
    MQTT_TOPIC_PREFIX="satellite" \
    MQTT_USERNAME="satellite" \
    SATELLITE_MQTT_PASSWORD="dummypassword" \
    HTTP_HOST="127.0.0.1" \
    HTTP_PORT="8000" \
    DEEPGRAM_API_KEY="" \
    LOG_LEVEL="INFO" \
    PYTHONUNBUFFERED="1"

# Expose RTP port and HTTP port
EXPOSE ${RTP_PORT}/udp
EXPOSE ${HTTP_PORT}

# Give the runtime only its private writable state directory.
RUN groupadd --gid 1001 satellite \
    && useradd --uid 1001 --gid 1001 --home-dir /var/lib/satellite-agent --create-home \
        --shell /usr/sbin/nologin satellite
USER 1001:1001

# Run the application
CMD ["python", "main.py"]
