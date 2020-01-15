#!/bin/bash

sudo docker run -ti --name dawn_eval --net=host --volume-driver=nvidia-docker \
                                    --volume nvidia_driver_418.67:/usr/local/nvidia:ro \
                                    --device /dev/nvidia0:/dev/nvidia0 \
                                    --device /dev/nvidia1:/dev/nvidia1 \
                                    --device /dev/nvidiactl:/dev/nvidiactl \
                                    --device /dev/nvidia-uvm:/dev/nvidia-uvm \
                                    --privileged=true -v /home/dawnbench_infer:/app \
                                    nvcr.io/nvidia/tensorrt:19.09-py3 bash
