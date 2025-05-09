docker compose -f Docker/serve/docker-compose-inference.yml down

DOCKER_BUILDKIT=1 docker compose -f Docker/serve/docker-compose-inference.yml build

docker compose -f Docker/serve/docker-compose-inference.yml up -d
