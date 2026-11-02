# Dockerfile for NeuroADHD RawMRI PoC V2
# Implements hardened execution environment with Python 3.11 on Debian Bookworm
FROM python:3.11-slim

# Prevent buffering of stdout/stderr and disable Python bytecode file generation
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLCONFIGDIR=/tmp/matplotlib \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2

# Create unprivileged service user and group
RUN groupadd -r neuro && useradd -r -g neuro -m -d /home/neuro neuro

# Set up working directory
WORKDIR /opt/neuro_adhd

# Install locked dependencies first to maximize build caching
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt

# Copy application source code, configs, scripts, and tests
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md .

# Create directory mounts and ensure permissions
RUN mkdir -p /opt/neuro_adhd/outputs /opt/neuro_adhd/data/atlases /tmp/matplotlib \
    && chown -R neuro:neuro /opt/neuro_adhd /tmp/matplotlib

# Switch to non-root user
USER neuro

# Default entrypoint runs the PoC V2 runner
ENTRYPOINT ["python", "scripts/run_poc_v2.py"]
CMD ["--help"]
