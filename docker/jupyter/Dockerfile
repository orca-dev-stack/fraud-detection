FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

# System deps
RUN apt update && apt install -y \
    python3 python3-pip python3-dev git \
    && rm -rf /var/lib/apt/lists/*

# Python deps (Jupyter + ML + ingestion)
RUN pip install --no-cache-dir \
    jupyterlab \
    numpy pandas scikit-learn matplotlib seaborn \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    kaggle \
    requests \
    google-cloud-bigquery \
    pyarrow \
    pyyaml

# Jupyter config (no token, remote access)
RUN mkdir -p /root/.jupyter
COPY jupyter/jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py

# Workdir is your project root
WORKDIR /workspace

# Copy project (optional; if you prefer volume mount, you can skip this)
COPY . /workspace

# Default command: Jupyter Lab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.token=''"]
