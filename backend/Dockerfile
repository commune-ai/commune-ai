# Use nvidia/cuda image
FROM node:alpine
FROM nvidia/cuda:11.2.0-base-ubuntu18.04
FROM python:3.8.2-slim

ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1

# Install linux dependencies
RUN apt-get update 
RUN apt-get install -y libssl-dev
RUN apt-get install -y npm
RUN apt-get install -y gcc

# RUN apt-get install python-tk python3-tk tk-dev

# install ganache-cli (TODO: run this in another container)
RUN npm install -g ganache-cli

WORKDIR /app

## -- Enter Virtual Env Installs our requirements into the Docker image
COPY ./requirements.txt .

RUN pip install -r requirements.txt
RUN pip install eth-brownie

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk

COPY ./scripts scripts
RUN chmod +x ./scripts/*

# set working directory


