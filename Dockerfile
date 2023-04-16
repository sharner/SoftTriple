ARG PYTORCH="22.11"

FROM nvcr.io/nvidia/pytorch:${PYTORCH}-py3

#RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
#    && cd onnx-tensorrt && mkdir build && cd build \
#    && cmake .. && make install \
#    && cd ../.. && rm -rf onnx-tensorrt

RUN TZ=America/Los_Angeles apt-get update && DEBIAN_FRONTEND=noninteractive \
apt-get install -y dialog apt-utils tzdata

RUN apt-get update && apt-get install -y \
	git \
	protobuf-compiler \
	python3-pil \
	chromium-chromedriver

COPY . /SoftTriple

RUN cd /SoftTriple \
    && python -m pip install --upgrade pip \
    && pip install --no-cache-dir -e . \
    && pip install -r requirements.txt

RUN curl -LO https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN apt-get install -y ./google-chrome-stable_current_amd64.deb
RUN rm google-chrome-stable_current_amd64.deb

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Other deps

RUN pip install --upgrade gensim

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
ENV LAYERJOT_HOME=/layerjot
