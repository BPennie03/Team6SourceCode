#!/bin/bash

OUTPUT_DIR="src/output"  # Adjusted to match Dockerfile and script
IMAGE_NAME="test-object-detection"
DOCKERFILE_PATH="."
CONTAINER_OUTPUT_DIR="/src/output"

# Build the Docker image
echo "Building Docker image..."
docker build -t $IMAGE_NAME $DOCKERFILE_PATH

# Create the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the Docker container with volume mapping
echo "Running Docker container..."
docker run -it --rm -v $(pwd)/$OUTPUT_DIR:$CONTAINER_OUTPUT_DIR $IMAGE_NAME
