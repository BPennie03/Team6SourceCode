# Test Object Detection

## Getting Started

To get started, follow the instructions below.

### Prerequisites
- [Docker](https://www.docker.com/get-started) installed on your machine.

## Running the Docker container
Run using `./run.sh` in terminal

### OR

Build the Docker Image:
```bash
docker build -t test-object-detection .
```

Run the Docker Container:
```bash
docker run -it --rm -v $(pwd)/src/output:/src/output test-object-detection
```