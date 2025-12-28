# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Install system dependencies
# We use generic names (without version numbers) to avoid "Unable to locate package"
RUN apt-get update && apt-get install -y \
    wget \
    tar \
    fontconfig \
    libfontconfig1 \
    libgraphite2-3 \
    libharfbuzz0b \
    libicu-dev \
    libssl-dev \
    zlib1g \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Use build-time argument to detect architecture
ARG TARGETARCH

# Install Tectonic based on architecture
RUN if [ "$TARGETARCH" = "amd64" ]; then \
        ARCH="x86_64-unknown-linux-musl"; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
        ARCH="aarch64-unknown-linux-musl"; \
    else \
        # Default to x86 if not specified
        ARCH="x86_64-unknown-linux-musl"; \
    fi && \
    wget "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic%400.15.0/tectonic-0.15.0-${ARCH}.tar.gz" && \
    tar -xzf "tectonic-0.15.0-${ARCH}.tar.gz" && \
    mv tectonic /usr/local/bin/ && \
    rm "tectonic-0.15.0-${ARCH}.tar.gz"

WORKDIR /app

# Ensure output directory exists for mounting
RUN mkdir -p /app/output

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "src/main.py"]
