# Set the base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y libgomp1 && apt-get clean

# Set working directory
WORKDIR /app

# Copy and install dependencies
COPY requirements-dockers.txt .
RUN pip install --upgrade pip && pip install -r requirements-dockers.txt

# Copy application and config files
COPY app.py .
COPY run_information.json .

# Create and copy preprocessor file
RUN mkdir -p data/processed
COPY data/processed/preprocessor.pkl data/processed/

# Create and copy raw data file
RUN mkdir -p data/raw
COPY data/raw/swiggy.csv data/raw/

# Copy utility modules
COPY notebooks/ notebooks/

# Ensure Python can find the notebooks directory
ENV PYTHONPATH="/app"

# Expose the app port
EXPOSE 8000

# Default command to run the app
CMD ["python", "app.py"]

