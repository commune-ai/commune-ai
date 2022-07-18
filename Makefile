kill_all:
	docker stop $(docker ps -a -q)

up:
	docker-compose up -d --remove-orphans

down:
	docker-compose down

restart:
	docker-compose down; docker-compose up -d --remove-orphans;



