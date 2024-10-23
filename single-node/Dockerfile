###################################################
# Initial image is just for building things

FROM nvcr.io/nvidia/pytorch:22.12-py3 AS build

# Install build dependencies
RUN apt update && \
    apt install -y python3.8-venv && \
    python3 -m pip install --upgrade build ninja cmake wheel pybind11

# Download the repositories needed
RUN git clone -b single-or-multi https://github.com/alan-turing-institute/neox-docker.git --recurse-submodules -j8 --depth 1

# Apply the patches to the submodules (applying only the single-node patch for gpt-neox)
RUN cd neox-docker/triton && \
    git apply ../patches/triton/*.patch
RUN cd neox-docker/gpt-neox && \
    git apply ../patches/gpt-neox/0001-fixes-to-v1-tag-for-baskerville-single-node-launchin.patch

# Build a version of Triton we can use
RUN cd neox-docker/triton/python && \
    python3 setup.py bdist_wheel && \
    python3 -m pip install dist/triton-0.4.2-cp38-cp38-linux_aarch64.whl

# Install dependencies
RUN cd neox-docker/gpt-neox/requirements && \
    pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-onebitadam.txt \
        -r requirements-sparseattention.txt
RUN pip install --no-cache-dir -v --disable-pip-version-check \
        --global-option="--cpp_ext" --global-option="--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597

# Build megatron
RUN python3 neox-docker/gpt-neox/megatron/fused_kernels/setup.py bdist_wheel && \
    python3 -m pip install dist/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl

###################################################
# This will be the final image

FROM nvcr.io/nvidia/pytorch:22.12-py3 AS deploy

# Copy in build artefacts from the build image
COPY --from=build /workspace/neox-docker/triton/python/dist/triton-0.4.2-cp38-cp38-linux_aarch64.whl /workspace/triton-0.4.2-cp38-cp38-linux_aarch64.whl
COPY --from=build /workspace/dist/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl /workspace/fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl
COPY --from=build /workspace/neox-docker/gpt-neox /gpt-neox

RUN python3 -m pip install triton-0.4.2-cp38-cp38-linux_aarch64.whl && \
    rm triton-0.4.2-cp38-cp38-linux_aarch64.whl

# Install dependencies
RUN cd /gpt-neox/requirements && \
    pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-onebitadam.txt \
        -r requirements-sparseattention.txt
RUN pip install --no-cache-dir -v --disable-pip-version-check \
        --global-option="--cpp_ext" --global-option="--cuda_ext" \
        git+https://github.com/NVIDIA/apex.git@a651e2c24ecf97cbf367fd3f330df36760e1c597
RUN python3 -m pip install fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl && \
    rm fused_kernels-0.0.1-cp38-cp38-linux_aarch64.whl

# Patch Deepspeed for single-node MPI launching
RUN sed -i \
        -e s/\'-hostfile\',/\#\'-hostfile\',/ \
        -e s/f\'\{self\.args\.hostfile\}\',/\#f\'\{self\.args\.hostfile\}\',/ \
        /usr/local/lib/python3.8/dist-packages/deepspeed/launcher/multinode_runner.py

# Set up execution environment
ENV LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/sklearn/__check_build/../../scikit_learn.libs/libgomp-d22c30c5.so.1.0.0
ENV OMP_NUM_THREADS=1

# Clear staging
RUN mkdir -p /tmp && chmod 0777 /tmp

WORKDIR /gpt-neox
