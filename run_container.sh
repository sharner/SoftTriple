#!/usr/bin/env bash

docker run \
	--gpus all \
	--mount type=bind,source="$(pwd)",target=/layerjot/SoftTriple \
	--mount type=bind,source="${LAYERJOT_HOME}/ljcv-serving",target=/layerjot/ljcv-serving \
	--mount type=bind,source="${N1_DATA}",target=/data \
	--rm --ipc=host -it soft-triple-n1
