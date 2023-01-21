#!/usr/bin/env bash

docker run \
	--gpus all \
	-v "$HOME/.config/gcloud":/gcp/config:ro \
	--env CLOUDSDK_CONFIG=/gcp/config \
	--env GOOGLE_APPLICATION_CREDENTIALS=/gcp/config/application_default_credentials.json \
	--mount type=bind,source="$(pwd)",target=/layerjot/SoftTriple \
	--mount type=bind,source="$(pwd)/..",target=/layerjot \
	--mount type=bind,source="/data/SoftTriple_Data",target="/data/SoftTriple_Data "\
	--mount type=bind,source="/data",target=/data \
	--rm --network=host -it softtriple.forest:latest
