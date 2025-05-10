# Use a slim Python base image to reduce size
FROM python:3.9.6-slim

# Set working directory
WORKDIR /cr-app

RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    python-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose Streamlit’s default port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "cr-app.py", "--server.port=8501", "--server.headless=true"]