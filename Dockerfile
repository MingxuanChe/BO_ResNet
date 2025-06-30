FROM python:3.10-slim

# Set working directory
WORKDIR /project

# Install dependencies
RUN apt-get update && apt-get install -y \
    libgmp-dev \
    libcdd-dev \
    cmake \
    g++ \
    make \
    python3-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir Pillow==10.0.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge

# Copy project
COPY . .

# Create results directory
RUN mkdir -p /project/results && chmod 755 /project/results

# Environment variables
ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1

# Define volume
VOLUME ["/project/results"]

# Run main script
CMD ["python", "main.py"]
