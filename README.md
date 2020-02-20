# Dawnbench Inference on Imagenet
## Summary
The inference speed of ResNet26-based model is improved through PAI-Blade. PAI-Blade is the framework for high-efficiency model deployment developed by AliCloud PAI: [PAI-Blade](https://help.aliyun.com/document_detail/145713.html). The BBS (Blade Benchmark Suite) can be accessed: [BBS](https://help.aliyun.com/document_detail/140559.html?spm=a2c4g.11186623.6.664.1137148cd2FacP). In addition, optimized INT8 conv2d operators are generated through [TVM TensorCore AutoCodeGen](https://docs.tvm.ai/tutorials/optimize/opt_matmul_auto_tensorcore.html?spm=ata.13261165.0.0.3eba2907mlPGq6).

## Run inference
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
4. Start nvcr.io/nvidia/tensorrt:19.09-py3 container, and install dependencies:
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
