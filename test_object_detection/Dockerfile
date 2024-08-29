# Use an official Python runtime as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file to the working directory
COPY requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app's source code to the working directory
COPY . .

# Set environment variable to match Python script paths
ENV PYTHONPATH=/src

EXPOSE 80

# Run the script
CMD ["python3", "src/object_detection.py"]
