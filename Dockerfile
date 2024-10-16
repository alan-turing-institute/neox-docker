###################################################
# Initial image is just for building things

FROM nvcr.io/nvidia/pytorch:22.12-py3 AS build

# Install build dependencies
RUN apt update && \
    apt install -y python3.8-venv && \
    python3 -m pip install --upgrade build ninja cmake wheel pybind11

# Build a version of Triton we can use
RUN git clone https://github.com/triton-lang/triton.git -b v0.4 --depth 1 && \
    sed -i -e s/version=\"0\.4\.0\"/version=\"0\.4\.2\"/ triton/python/setup.py && \
    pushd triton/python && \
    python3 setup.py bdist_wheel && \
    popd && \
    python3 -m pip install triton/python/dist/triton-0.4.2-cp38-cp38-linux_aarch64.whl

# Collect and patch the gpt-neox files
RUN git clone https://github.com/edchapman88/gpt-neox-v1-fixed.git -b fix --depth 1 && \
    sed -i -e s/tiktoken==0\.1\.2/tiktoken==0\.4\.0/ gpt-neox-v1-fixed/requirements/requirements.txt && \
    printf "best-download\n" >> gpt-neox-v1-fixed/requirements/requirements.txt && \
    sed -i -e s/cupy-cuda111==8\.6\.0/cupy-cuda11x==12\.3\.0/ gpt-neox-v1-fixed/requirements/requirements-onebitadam.txt

# Install dependencies
RUN pip install --no-cache-dir \
        -r gpt-neox-v1-fixed/requirements/requirements.txt \
        -r gpt-neox-v1-fixed/requirements/requirements-onebitadam.txt \
        -r gpt-neox-v1-fixed/requirements/requirements-sparseattention.txt \
        protobuf==3.20.1
RUN pip install --no-cache-dir -v --disable-pip-version-check \
        --global-option="--cpp_ext" --global-option="--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

# Build megatron
RUN python3 gpt-neox-v1-fixed/megatron/fused_kernels/setup.py bdist_wheel && \
    python3 -m pip install dist/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl

###################################################
# This will be the final image

FROM nvcr.io/nvidia/pytorch:22.12-py3 AS deploy

# Copy in build artefacts from the build image
COPY --from=build /workspace/triton/python/dist/triton-0.4.2-cp38-cp38-linux_aarch64.whl /workspace/triton-0.4.2-cp38-cp38-linux_aarch64.whl
COPY --from=build /workspace/dist/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl /workspace/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl
COPY --from=build /workspace/gpt-neox-v1-fixed /gpt-neox

RUN python3 -m pip install triton-0.4.2-cp38-cp38-linux_aarch64.whl && \
    rm triton-0.4.2-cp38-cp38-linux_aarch64.whl

# Install dependencies
RUN pip install --no-cache-dir \
        -r /gpt-neox/requirements/requirements.txt \
        -r /gpt-neox/requirements/requirements-onebitadam.txt \
        -r /gpt-neox/requirements/requirements-sparseattention.txt \
        protobuf==3.20.1
RUN pip install --no-cache-dir -v --disable-pip-version-check \
        --global-option="--cpp_ext" --global-option="--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597
RUN python3 -m pip install fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl && \
    rm fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl

# Patch Deepspeed for MPI
RUN sed -i \
        -e s/\'-hostfile\',/\#\'-hostfile\',/ \
        -e s/f\'\{self\.args\.hostfile\}\',/\#f\'\{self\.args\.hostfile\}\',/ \
        /usr/local/lib/python3.8/dist-packages/deepspeed/launcher/multinode_runner.py

# Set up execution environment
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV OPAL_PREFIX=/opt/hpcx/ompi
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

WORKDIR /gpt-neox
