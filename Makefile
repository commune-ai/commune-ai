kill_all:
	 docker kill $(docker ps -q)

up:
	docker compose up -d --remove-orphans
down:
	docker-compose down
restart:
	docker-compose down; docker-compose up -d --remove-orphans;
setup_env:
	docker exec -it $ (arg) bash
logs:
	docker logs ${arg}
frontend:
	docker exec -it frontend bash

speedtest:
	curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -
