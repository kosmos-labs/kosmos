FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    htop \
    iotop \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    psutil \
    GPUtil \
    numpy

# Set working directory
WORKDIR /workspace

# Copy training script
COPY train.py .

# Create output directory
RUN mkdir -p /workspace/output

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python3", "train.py"]