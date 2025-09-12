# Makefile for Garmin R10 Dashboard

# Variables
IMAGE_NAME = garmin-r10-dashboard
REGISTRY = ghcr.io/rpetersson
FULL_IMAGE = $(REGISTRY)/$(IMAGE_NAME)
TAG = latest
PORT = 8501

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  build       - Build the Docker image locally"
	@echo "  pull        - Pull image from GitHub Container Registry"
	@echo "  run         - Run the container"
	@echo "  stop        - Stop and remove the container"
	@echo "  logs        - Show container logs"
	@echo "  shell       - Get a shell inside the container"
	@echo "  clean       - Remove images and containers"
	@echo "  dev         - Run in development mode"
	@echo "  compose     - Run with docker-compose (uses GitHub registry image)"
	@echo "  compose-dev - Run with docker-compose (builds locally)"
	@echo "  push        - Push image to GitHub Container Registry"
	@echo "  login       - Login to GitHub Container Registry"

# Build the Docker image locally
.PHONY: build
build:
	docker build -t $(IMAGE_NAME):$(TAG) -t $(FULL_IMAGE):$(TAG) .

# Pull image from GitHub Container Registry
.PHONY: pull
pull:
	docker pull $(FULL_IMAGE):$(TAG)

# Login to GitHub Container Registry
.PHONY: login
login:
	@echo "Login to GitHub Container Registry with your GitHub token:"
	@echo "docker login ghcr.io -u <username> -p <token>"
	docker login ghcr.io

# Push image to GitHub Container Registry
.PHONY: push
push: build
	docker push $(FULL_IMAGE):$(TAG)

# Run the container (using GitHub registry image)
.PHONY: run
run:
	docker run -d --name $(IMAGE_NAME) -p $(PORT):$(PORT) -v $(PWD)/data:/app/data $(FULL_IMAGE):$(TAG)

# Run the container (using local image)
.PHONY: run-local
run-local:
	docker run -d --name $(IMAGE_NAME) -p $(PORT):$(PORT) -v $(PWD)/data:/app/data $(IMAGE_NAME):$(TAG)

# Stop and remove the container
.PHONY: stop
stop:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true

# Show container logs
.PHONY: logs
logs:
	docker logs -f $(IMAGE_NAME)

# Get a shell inside the container
.PHONY: shell
shell:
	docker exec -it $(IMAGE_NAME) /bin/bash

# Clean up images and containers
.PHONY: clean
clean:
	docker stop $(IMAGE_NAME) || true
	docker rm $(IMAGE_NAME) || true
	docker rmi $(IMAGE_NAME):$(TAG) || true
	docker rmi $(FULL_IMAGE):$(TAG) || true

# Run in development mode (with volume mount for live reloading)
.PHONY: dev
dev:
	docker run -it --rm --name $(IMAGE_NAME)-dev -p $(PORT):$(PORT) \
		-v $(PWD):/app -v $(PWD)/data:/app/data \
		$(IMAGE_NAME):$(TAG) \
		streamlit run app.py --server.address 0.0.0.0 --server.port $(PORT) --server.fileWatcherType poll

# Run with docker-compose (uses GitHub registry image)
.PHONY: compose
compose:
	docker-compose up -d

# Run with docker-compose (builds locally)
.PHONY: compose-dev
compose-dev:
	# Temporarily use local build
	sed -i.bak 's/^    image:/#    image:/' docker-compose.yml && \
	sed -i.bak 's/^    # build:/    build:/' docker-compose.yml && \
	sed -i.bak 's/^    #   context:/      context:/' docker-compose.yml && \
	sed -i.bak 's/^    #   dockerfile:/      dockerfile:/' docker-compose.yml && \
	docker-compose up --build -d && \
	mv docker-compose.yml.bak docker-compose.yml

# Stop docker-compose
.PHONY: compose-down
compose-down:
	docker-compose down

# Test the container
.PHONY: test
test: 
	@echo "Testing container..."
	docker run --rm --name $(IMAGE_NAME)-test -p 8502:8501 -d $(FULL_IMAGE):$(TAG)
	@echo "Waiting for container to start..."
	sleep 15
	@echo "Testing health endpoint..."
	curl -f http://localhost:8502/_stcore/health || (docker stop $(IMAGE_NAME)-test && exit 1)
	@echo "Test passed!"
	docker stop $(IMAGE_NAME)-test
