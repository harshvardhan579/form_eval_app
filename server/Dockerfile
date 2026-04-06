FROM python:3.9-slim

WORKDIR /app

# Install dependencies (Copy only requirements.txt first for better Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Railway provides a dynamic PORT environment variable.
# We expose it but the actual binding occurs in the command below.
ENV PORT=8000
EXPOSE ${PORT}

# Command to run uvicorn on the container, reading PORT from environment
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
