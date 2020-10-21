FROM nvcr.io/nvidia/pytorch:20.09-py3

RUN git clone --recurse-submodules https://github.com/onnx/onnx-tensorrt.git \
    && cd onnx-tensorrt && mkdir build && cd build \
    && cmake .. && make install \
    && cd ../.. && rm -rf onnx-tensorrt

COPY . /workspace
RUN pip install --no-cache-dir -e softriple/
