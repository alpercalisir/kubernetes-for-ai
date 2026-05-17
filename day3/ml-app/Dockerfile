# Stage 1: Builder
FROM python:3.11 AS builder
WORKDIR /install
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt && \
    find /install/packages -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "test" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -name "*.pyc" -delete && \
    find /install/packages -name "*.pyo" -delete && \
    find /install/packages -name "*.pyd" -delete && \
    find /install/packages -type d -name "*.dist-info" \
        ! -name "werkzeug-*" \
        ! -name "flask-*" \
        ! -name "importlib_metadata-*" \
        -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "*.egg-info" \
        ! -name "werkzeug*" \
        ! -name "flask*" \
        -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "examples" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "docs" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "doc" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -type d -name "data" -exec rm -rf {} + 2>/dev/null || true && \
    find /install/packages -name "*.pyx" -delete && \
    find /install/packages -name "*.pxd" -delete && \
    find /install/packages -name "*.c" -delete 2>/dev/null || true && \
    find /install/packages -name "*.so" -exec strip --strip-all {} \; 2>/dev/null || true && \
    find /install/packages -type d -name "*.dist-info" -exec sh -c \
        'find "$1" -type f ! -name "METADATA" ! -name "WHEEL" ! -name "RECORD" \
        ! -name "entry_points.txt" ! -name "top_level.txt" -delete' _ {} \; 2>/dev/null || true
# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /install/packages /usr/local
COPY app.py .
COPY model.pkl .
ENV PYTHONUNBUFFERED=1
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser
EXPOSE 8000
CMD ["python", "app.py"]