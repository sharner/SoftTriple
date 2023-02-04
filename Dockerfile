FROM nvcr.io/nvidia/pytorch:22.10-py3
#FROM nvcr.io/nvidia/pytorch:21.09-py3

#RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
#    && cd onnx-tensorrt && mkdir build && cd build \
#    && cmake .. && make install \
#    && cd ../.. && rm -rf onnx-tensorrt

COPY . /SoftTriple
RUN cd /SoftTriple && \
    pip install --no-cache-dir -e .
RUN pip install timm altair duckdb
RUN pip install gcsfs

# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
  && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
  && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin
ENV LAYERJOT_HOME=/layerjot
