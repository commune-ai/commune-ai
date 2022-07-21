
kill_all:
	 docker kill $(docker ps -q)

up:
	docker compose up -d --remove-orphans
down:
	docker-compose down
restart:
	docker-compose down; docker-compose up -d --remove-orphans;
env:
	docker exec -it $(arg) bash
logs:
	docker logs ${arg}
frontend:
	docker exec -it frontend bash