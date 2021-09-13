#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/SoftTriple \
	--mount type=bind,source="${LAYERJOT_HOME}/SoftTriple_Data",target=/layerjot/DataModels \
	--mount type=bind,source="/data",target=/data \
	--rm --ipc=host -it soft-triple-n1:latest
