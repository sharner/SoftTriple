#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/SoftTriple \
	--mount type=bind,source="/home/data",target=/data \
	--rm --shm-size 2G --network=host -it softtriple:latest
