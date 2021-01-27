from pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN apt-get install -y python-pip && apt-get install wget

RUN mkdir -p /opt/colorize_flask/
COPY colorize_flask /opt/colorize_flask/

RUN mkdir -p /opt/colorize_flask/model/ \
&& mkdir -p /opt/colorize_flask/static/upload/

RUN cd /opt/colorize_flask \
 && pip install Flask \
 && pip install scikit-image
RUN cd /opt/colorize_flask/model \
 && wget https://www.dropbox.com/s/w5sj7hiy5um4fyx/gen.pt \
 && cd /opt/colorize_flask/
WORKDIR /opt/colorize_flask/

CMD ["colorize.py"]
ENTRYPOINT ["python"]

