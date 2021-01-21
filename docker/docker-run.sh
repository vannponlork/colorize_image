#dicm#!/bin/bash
docker run -d -it -p 5000:5000 --name colorize colorize:latest
#docker run -d -it -p 5000:5000 --gpus all --name colorize --mount type=bind,source=`pwd`/../colorize_flask/,target=/opt/colorize_flask/ colorize:latest /bin/bash

#docker run --name=colorize -v "$(pwd)"/colorize_flask:/opt/colorize_flask/ -p 5000:5000 colorize:latest