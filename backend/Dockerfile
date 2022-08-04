# Use nvidia/cuda image
FROM nvidia/cuda:11.2.1-base


ENV PYTHONUNBUFFERED 1
ENV PYTHONDONTWRITEBYTECODE 1
ENV TZ=Europe/London
ARG DEBIAN_FRONTEND=noninteractive

#nvidia key migration
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y apt-utils curl git cmake build-essential unzip python3-pip  wget iproute2 software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install python3 python3-dev -y
RUN python3 -m pip install --upgrade pip

# Install linux dependencies
RUN apt-get install -y libssl-dev
RUN apt-get install -y gcc
RUN apt-get install -y npm

# install ganache for simulating evm vm
RUN npm install -g ganache-cli



WORKDIR /app
RUN pip3 install --upgrade pip

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./ipfsspec/ /app/ipfsspec/
RUN pip install -e ./ipfsspec
# RUNNING THESE IN THE EXECUTE COMMAND 


# add BITTENSOR commune fork
ADD ./bittensor /app/bittensor
RUN pip install -e ./bittensor

# TEMPORARY UNTIL MERGE WITH FSSPEC add ipfsspec



# RUN pip install brownie
# ADD ./brownie/ /app/brownie/
# RUN pip install -e ./brownie


COPY ./scripts scripts
RUN chmod +x ./scripts/*

RUN python3 -m  pip install "ray[serve]"

RUN alias python=python3
RUN pip install --upgrade fastapi

# COPY --from=library/docker:latest /usr/local/bin/docker /usr/bin/docker
# COPY --from=docker/compose:latest /usr/local/bin/docker-compose /usr/bin/docker-compose
