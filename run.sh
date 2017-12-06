#!/bin/bash

if [[ ! -z  `which nvidia-docker`  ]]
then
    DOCKER_CMD=nvidia-docker
elif [[ ! -z  `which docker`  ]]
then
    echo "WARNING: nvidia-docker not found. Nvidia drivers may not work." >&2
    DOCKER_CMD=docker
else
     echo "ERROR: docker or nvidia-docker not found. Aborting." >&2
    exit 1
fi

DIRECTORY=vizdoom_doubledqn

if [ -e temp ]; then
  rm temp -rf
fi

${DOCKER_CMD} run -ti --net=host --privileged -e DISPLAY=${DISPLAY} --name ${DIRECTORY} ${DIRECTORY}
$DOCKER_CMD cp ${DIRECTORY}:/home/vizdoom/ ./temp/
$DOCKER_CMD rm ${DIRECTORY}