#!/bin/bash
docker build --file="Dockerfile" --tag="colorize:$(date +%Y%m%d%H%M%S)" --tag="colorize" ..
