FROM python:3.12-slim AS base

WORKDIR /app

# Install system deps for scipy/numba
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY proto/ proto/
COPY scripts/ scripts/
COPY src/ src/

# Install grpcio-tools for proto compilation only, then compile and uninstall
RUN pip install --no-cache-dir "grpcio-tools>=1.60" && \
    mkdir -p src/serialization/proto && \
    python -m grpc_tools.protoc \
    -Iproto \
    --python_out=src/serialization/proto \
    --pyi_out=src/serialization/proto \
    proto/trajectory.proto && \
    touch src/serialization/proto/__init__.py && \
    pip uninstall -y grpcio-tools || true

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
