FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (fixed formatting)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit Cloud Run settings
EXPOSE 8080
ENV PORT=8080

# Streamlit entry point
CMD ["sh", "-c", "streamlit run main.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS false --server.headless true"]