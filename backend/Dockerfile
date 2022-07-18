# Use nvidia/cuda image
FROM node:alpine
FROM nvidia/cuda:11.2.1-base




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


#nvidia key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3-pip  wget iproute2 software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3 python3-dev python3-tk
RUN python3 -m pip install --upgrade pip
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
## -- Enter Virtual Env Installs our requirements into the Docker image
COPY ./requirements.txt .
RUN pip install -r requirements.txt


COPY ./scripts scripts
RUN chmod +x ./scripts/*

# set working directory


