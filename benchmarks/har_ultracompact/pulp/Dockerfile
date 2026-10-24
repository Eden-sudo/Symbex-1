# The PULP environment of Daghero et al. (CF 2021): the RISC-V PULP compiler + the PULP SDK + the GVSoC simulator.
#
#   docker build -t symbex-pulp .                                      # once (toolchain download + GVSoC build)
#   docker run --rm -it -v ~/Proyectos:/work symbex-pulp               # a shell with everything ready
#
# GVSoC is no longer inside the PULP SDK: it is its own repository (github.com/gvsoc/gvsoc), built with its
# submodules for the pulp-open target (its fabric controller is a RI5CY core, the one the ultracompactBNN templates run
# on: PULP_APP_FC_SRCS). It needs a recent gcc and cmake (its authors use gcc 11.2, cmake 3.18): Ubuntu 22.04.
FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git curl wget ca-certificates bzip2 cmake scons rsync autoconf automake texinfo libtool \
        pkg-config doxygen python3 python3-pip python3-dev libsdl2-dev libsdl2-ttf-dev libsndfile1-dev libftdi-dev \
        libftdi1 libusb-1.0-0-dev gtkwave device-tree-compiler libelf-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install argcomplete pyelftools six prettytable pyyaml

# the precompiled PULP RISC-V GCC (release v1.0.16). If the download fails, open
# https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/tag/v1.0.16 and put the exact name of the
# Ubuntu tarball in TOOLCHAIN_TAR.
ARG TOOLCHAIN_TAR=v1.0.16-pulp-riscv-gcc-ubuntu-18.tar.bz2
WORKDIR /opt
RUN wget -q https://github.com/pulp-platform/pulp-riscv-gnu-toolchain/releases/download/v1.0.16/${TOOLCHAIN_TAR} \
    && tar xjf ${TOOLCHAIN_TAR} && rm ${TOOLCHAIN_TAR} \
    && ln -s /opt/$(ls /opt | grep -m1 pulp-riscv-gcc) /opt/pulp-gcc
ENV PULP_RISCV_GCC_TOOLCHAIN=/opt/pulp-gcc
ENV PATH=/opt/pulp-gcc/bin:$PATH

# the PULP SDK (the runtime, the build rules)
RUN git clone https://github.com/pulp-platform/pulp-sdk.git /opt/pulp-sdk

# GVSoC, with its submodules, for pulp-open
RUN git clone --recursive https://github.com/gvsoc/gvsoc.git /opt/gvsoc
WORKDIR /opt/gvsoc
RUN for r in $(find . -maxdepth 3 -name requirements.txt); do pip3 install -r $r || true; done
RUN make all TARGETS=pulp-open
ENV PATH=/opt/gvsoc/install/bin:$PATH

RUN echo 'source /opt/pulp-sdk/configs/pulp-open.sh' >> /root/.bashrc \
    && echo 'export PATH=/opt/gvsoc/install/bin:$PATH' >> /root/.bashrc
WORKDIR /work
CMD ["bash"]
