#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

# Export CI_TOKEN (set your token here)
export CI_TOKEN=94f5f599e01d773416c043c7b7e819b789f3b7b8

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 565393027942.dkr.ecr.ap-southeast-2.amazonaws.com/swiggy-food-delivery-time-prediction:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=swiggy-food-delivery-time-prediction)" ]; then
    echo "Stopping existing container..."
    docker stop swiggy-food-delivery-time-prediction
fi

if [ "$(docker ps -aq -f name=swiggy-food-delivery-time-prediction)" ]; then
    echo "Removing existing container..."
    docker rm swiggy-food-delivery-time-prediction
fi

echo "Starting new container..."
docker run -d -p 80:8000 --name swiggy-food-delivery-time-prediction \
  -e DAGSHUB_USER_TOKEN=$CI_TOKEN \
  565393027942.dkr.ecr.ap-southeast-2.amazonaws.com/swiggy-food-delivery-time-prediction:latest

echo "Container started successfully."
