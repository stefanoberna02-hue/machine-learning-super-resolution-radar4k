# -------------------------------------------------------------------
# Base image used by the authors (RadarHD era, ~2022).
# Equivalent to "docker run pytorch/pytorch ..." from README,
# but with the exact version that matches their environment.
# -------------------------------------------------------------------
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# -------------------------------------------------------------------
# Timezone: authors ask user to select "US Eastern" manually.
# We automate this step.
# -------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
    apt-get install -y tzdata && \
    dpkg-reconfigure --frontend noninteractive tzdata


RUN apt-get update && apt-get install -y build-essential

RUN pip install pillow \
 && pip install torchsummary \
 && pip install scipy \
 && pip install pandas \
 && pip install open3d 
 
RUN pip install plyfile
RUN pip install tensorboard


WORKDIR /project-2-radar4k
COPY ./ .