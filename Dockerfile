FROM python:3.12-slim

WORKDIR /app

# Install dependencies first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY tools/ tools/
COPY utils/ utils/
COPY wav/ wav/
COPY audio/ audio/

# Run the app
CMD ["python", "app.py"]
