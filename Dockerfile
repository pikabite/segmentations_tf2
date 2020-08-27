FROM tensorflow/tensorflow:2.2.0-gpu-jupyter

RUN mkdir /dy
WORKDIR /dy
EXPOSE 15011

ENV DOCKERNAME=segmentation_tf2

RUN apt update \
# && apt install -y python3 \
# && apt install -y python3-pip git \
&& pip install tqdm pillow pyyaml scipy \
&& apt install -y libsm6 libxext6 libxrender1 libfontconfig1 \
&& pip3 install opencv-python opencv-contrib-python 
RUN pip3 install -U git+https://github.com/albu/albumentations
RUN python3 -m pip install --upgrade pip

# RUN pip3 install gast==0.2.2
# RUN pip3 install tensorflow-gpu==2.2.0

RUN useradd -ms /bin/bash kdy
USER kdy

ENTRYPOINT ["/bin/sh", "-c", "jupyter-notebook --no-browser --port 15011 --ip=0.0.0.0 --NotebookApp.token='kdy' --allow-root --NotebookApp.password='' --NotebookApp.allow_origin='*'"]

