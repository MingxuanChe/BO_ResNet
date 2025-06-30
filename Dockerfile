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
    git

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Environment variables
ENV PYTHONPATH=/project
ENV PYTHONUNBUFFERED=1

VOLUME ["/project/results_docker"]

# run main script
CMD ["python", "main.py"]
