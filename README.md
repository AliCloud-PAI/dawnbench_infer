# Dawnbench Inference on Imagenet

## run inference
1. Clone this repo.
2. Put Imagenet validation set (50000 images) in `imagenet/val_data/`.
```Example:
./imagenet/val_data/
├── ILSVRC2012_val_00000001.JPEG
├── ILSVRC2012_val_00000002.JPEG
├── ILSVRC2012_val_00000003.JPEG
├── ILSVRC2012_val_00000004.JPEG
├── ILSVRC2012_val_00000005.JPEG
├── ILSVRC2012_val_00000006.JPEG
...
```
3. Pull nvcr.io/nvidia/tensorrt:19.09-py3 from NGC.
4. Start nvcr.io/nvidia/tensorrt:19.09-py3 container, and install dependencies.
```
# start docker container
sh run.sh

# Install TensorFlow
pip install tensorflow==1.13.1
```
5. Set environment in host:
```
sh set_env.sh
```
6. In the container workspace, run cmd as below:
```
# Assuming the workspace is mounted as /app
cd /app

# Run for inference
sh eval.sh
```
