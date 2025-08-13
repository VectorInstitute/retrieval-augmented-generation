FROM ubuntu:22.04

EXPOSE 8888

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    sudo \
    curl \
    git \
    jq \
    tar \
    unzip \
    python3 \
    python3-pip \
    python3-venv \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# !!IMPORTANT!!
# THIS SECTION SHOULD NOT BE MODIFIED AS
# IT IS USED TO MAKE THIS IMAGE COMPATIBLE WITH CODER
#######################################################################
ARG USER=coder
RUN useradd --groups sudo --no-create-home --shell /bin/bash ${USER} \
    && echo "${USER} ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/${USER} \
    && chmod 0440 /etc/sudoers.d/${USER}

USER ${USER}
WORKDIR /home/${USER}
########################################################################

# Copy the code into the container
COPY --chown=${USER}:${USER} . /home/${USER}/retrieval-augmented-generation

# Set the working directory
WORKDIR /home/${USER}/retrieval-augmented-generation

# Start the container and run the project setup script
CMD ["bash", "scripts/setup.sh"]
