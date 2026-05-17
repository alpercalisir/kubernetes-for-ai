# Stage 1: Builder
FROM python:3.11 AS builder

WORKDIR /install

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install/packages -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages
COPY --from=builder /install/packages /usr/local

# Copy app files
COPY app.py .
COPY model.pkl .

ENV PYTHONUNBUFFERED=1

# Non-root user
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["python", "app.py"]