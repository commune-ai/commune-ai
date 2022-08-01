kill_all:
	 docker kill $(docker ps -q) && docker rm $(docker ps -q)

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

bash:
	docker exec -it ${arg} bash

sh:
	docker exec -it ${arg} bash
	
frontend:
	docker exec -it frontend bash

start:
	docker compose up ${arg} -d 

	
purge: 
	docker compose stop ${arg}; docker compose rm ${args};

speedtest:
	curl -s https://raw.githubusercontent.com/sivel/speedtest-cli/master/speedtest.py | python -


api:
	python3 commune/api/main.py 