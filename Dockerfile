FROM nvcr.io/nvidia/pytorch:22.10-py3
#FROM nvcr.io/nvidia/pytorch:21.09-py3

#RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
#    && cd onnx-tensorrt && mkdir build && cd build \
#    && cmake .. && make install \
#    && cd ../.. && rm -rf onnx-tensorrt

COPY . /SoftTriple
RUN cd /SoftTriple && \
    pip install --no-cache-dir -e .
RUN pip install timm
