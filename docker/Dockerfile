from pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
# RUN apt-get update && apt-get upgrade -y && apt-get install -y python3-pip python3-venv libsm6 libxext6 libxrender-dev

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN mkdir -p /opt/colorize_flask/

# COPY docker/requirements.txt /opt/lastmile/requirements.txt
RUN apt-get install -y python-pip

RUN cd /opt/colorize_flask \
 && pip install flask \
 && pip install Flask \
 && pip install scikit-image

COPY colorize_flask /opt/colorize_flask/
WORKDIR /opt/colorize_flask/

CMD ["hello.py"]
ENTRYPOINT ["python"]


