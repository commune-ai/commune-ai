kill_all:
	c=$(docker ps -q) && [[ $c ]] && docker kill $c

up:
	docker-compose up -d --remove-orphans

down:
	docker-compose down
restart:
	docker-compose down; docker-compose up -d --remove-orphans;

backend:
	docker exec -it backend bash

frontend:
	docker exec -it frontend bash