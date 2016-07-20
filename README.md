GPU Benchmark
=============

### mxnet

使用 `Dockerfile` 生成 docker image, 例如叫 `gpu_test`, 测试命令

`nvidia-docker run gpu_test python gpu_test/mxnet_cnn_test.py --data-dir cifar/ --lr 0.05 --lr-factor 0.94 --num-epoch 5 --gpus 0`

如果有多个 GPU, 则按如下形式修改相关参数 `--gpus 0,1,2`

### tensorflow

使用 `Dockerfile_tensorflow` 生成 docker image, 例如叫 `tf_gpu_test`, 测试命令

`nvidia-docker run tf_gpu_test python /tensorflow/tensorflow/models/image/cifar10/cifar10_multi_gpu_train.py --data_dir . --max_steps 10000 --num_gpus 1`

如有多个 GPU, 比如想测试 2 个, 则修改参数为 `--num_gpus 2`


- [Nvidia Docker](https://github.com/NVIDIA/nvidia-docker)
- [MXNET Docker](https://github.com/dmlc/mxnet/tree/master/docker)
- [TensorFlow Docker](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/tools/docker)
