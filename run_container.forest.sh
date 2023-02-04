#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/SoftTriple \
	--mount type=bind,source="/home/data",target=/data \
	--rm --ipc=host -it softtriple:latest
