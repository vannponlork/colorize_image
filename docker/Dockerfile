from pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN mkdir -p /opt/colorize_flask/
RUN mkdir -p /opt/colorize_flask/model/
RUN mkdir -p /opt/colorize_flask/static/upload/


RUN apt-get install -y python-pip && apt-get install wget

RUN cd /opt/colorize_flask \
 && pip install Flask \
 && pip install scikit-image
RUN cd /opt/colorize_flask/model \
 && wget https://uc0fa4e82e04492017e7e5e18ba8.dl.dropboxusercontent.com/cd/0/get/BHyqi8AyJSJ96NnAD0KaAgoc4nCmQTeQbdJS78VaqqkblHMq6WzCeDmT7Y7XysUpipX50plktUiXSxwVccyaTKR7yN6VhDxWEsxLX4HJ-9oeDr4ByNYLhGaoniPqNL6-p2U/file# \
 && cd /opt/colorize_flask/

COPY colorize_flask /opt/colorize_flask/
WORKDIR /opt/colorize_flask/

CMD ["colorize.py"]
ENTRYPOINT ["python"]

