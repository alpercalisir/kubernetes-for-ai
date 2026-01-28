# Stage 1: Builder
FROM python:3.11 AS builder

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /root/.local /root/.local

# Copy application files
COPY app.py .
COPY model.pkl .

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH

# Don't buffer Python output
ENV PYTHONUNBUFFERED=1

# Configure default port
ENV PORT=8000

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "app.py"]