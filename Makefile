kill_all:
	 docker kill $(docker ps -q)

up:
	docker compose up -d --remove-orphans

down:
	docker-compose down
restart:
	docker-compose down; docker-compose up -d --remove-orphans;
backend_env:
	docker exec -it backend bash

frontend:
	docker exec -it frontend bash