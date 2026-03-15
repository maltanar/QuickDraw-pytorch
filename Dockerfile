FROM rocm/pytorch:latest

# Set environment variables for better compatibility
ENV DEBIAN_FRONTEND=noninteractive

# Update and install system dependencies (if any are needed by ONNX/Brevitas)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install extra dependencies
# Using git+https for brevitas dev branch as per current setup in run_experiments.sh
RUN pip install --no-cache-dir \
    git+https://github.com/Xilinx/brevitas.git@dev \
    onnx \
    onnxscript \
    qonnx \
    onnxoptimizer

# Set default command
CMD ["bash"]
