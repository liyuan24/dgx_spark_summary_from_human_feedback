FROM nvcr.io/nvidia/pytorch:25.09-py3

ARG HF_TOKEN

RUN pip install streamlit timm wandb && \
    hf auth login --token $HF_TOKEN

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda-13.0/
ENV CUDA_PATH=$CUDA_HOME
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
ENV C_INCLUDE_PATH=$CUDA_HOME/include:$C_INCLUDE_PATH
ENV CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH

# install triton from source for latest blackwell support
RUN git clone https://github.com/triton-lang/triton.git && \
    cd triton && \
    git checkout c5d671f91d90f40900027382f98b17a3e04045f6 && \
    pip install -r python/requirements.txt && \
    pip install . && \
    cd ..

# install xformers from source for blackwell support
RUN git clone https://github.com/facebookresearch/xformers && \
    cd xformers && \
    git checkout 5146f2ab37b2163985c19fb4e8fbf6183e82f8ce && \
    git submodule update --init --recursive && \
    export TORCH_CUDA_ARCH_LIST="12.1" && \
    python setup.py install && \
    cd ..

RUN pip install unsloth==2025.9.11 unsloth_zoo==2025.9.14 bitsandbytes==0.48.0
# the default command is to start a bash shell
CMD ["/bin/bash"]

