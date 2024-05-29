# Use an official image as a base
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install --no-install-recommends -y ca-certificates gpg nano wget gcc g++ build-essential libxml2 libxml2-utils libxml2-dev git curl

# Install CMake from Kitware
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null  && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main' | tee /etc/apt/sources.list.d/kitware.list >/dev/null  && \
    apt-get update  && \
    apt-get install --no-install-recommends -y kitware-archive-keyring  && \
    echo 'deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal-rc main' | tee -a /etc/apt/sources.list.d/kitware.list >/dev/null  && \
    apt-get update  && \
    apt-get install --no-install-recommends -y cmake  && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /miniconda && \
    rm miniconda.sh

# Add Miniconda to PATH
ENV PATH="/miniconda/bin:${PATH}"

# Copy your environment file to the Docker image
COPY spconv.yml /setup/spconv.yml

# Install mamba as it is much faster than conda
RUN conda install -n base -c conda-forge mamba -y && \
    conda config --set solver libmamba

# Create a Conda environment based on the environment.yml file
RUN mamba env create -f /setup/spconv.yml

# Optional: Cleanup to reduce image size
RUN conda clean -a -y

# Make sure the environment is activated when the container runs
SHELL ["conda", "run", "-n", "spconv", "/bin/bash", "-c"]

# This is to ensure that the Conda environment is activated when you run the Docker container
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "spconv"]

