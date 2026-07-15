# profile

## setup

clone仓库之后，准备构建docker镜像。

```bash
docker build -f docker/Dockerfile.python3.12 . -t ${YOUR_NAME}-fastvideo:cu128
```

这个dockerfile和原仓库的有一点区别（去掉了编译内核的部分，需要启动容器后手动编译）。

打包好镜像后，在项目根目录下执行：

```bash
docker run --gpus all -it -d  --name ${YOUR_NAME}-fastvideo-cu128 --network host \
    --ipc=host\
    -v $(pwd):/FastVideo\
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -w /FastVideo \
    --cap-add=SYS_ADMIN \
    fastvideo:cu128
```

这个命令中，`--cap-add=SYS_ADMIN`和NCU有关，一些full模式下的stats需要管理员权限，或者服务器上有特殊配置的情况下可以不用`--cap-add=SYS_ADMIN`

进入容器后，处于`/FastVideo`路径下，执行：

```bash
docker exec -it ${YOUR_NAME}-fastvideo-cu128 /bin/bash

cd fastvideo-kernel
git submodule update --init --recursive
MAX_JOBS=8 ./build.sh

cd ..
```

这个步骤除了install fastvideo-kernel外，脚本还会安装ninja（如果需要反复修改FA的话，这一步安装的ninja很重要）。

验证环境：
1. 可以尝试`python examples/inference/basic/basic_self_forcing_causal.py`，这会下载一整个模型，这样一定能确定fastvideo是正确安装的
2. 或者`python examples/inference/profiling/profile_single_transformer_block.py`，这是跑单个transformer block

## profile 

### 完整生成过程

```bash
python examples/inference/profiling/basic_self_forcing_causal_profiling.py
```

这会执行一次完整的AR-diffusion过程，可以修改代码以获取视频产物。

命令可选参数：

`--bs`设置batch size，`--fsdp`开fsdp，`--num_gpus`指定运行GPU的数量。

### 单个DiT block

我最早写的是`examples/inference/profiling/profile_single_transformer_block.py`，这个文件构造了一个固定的视频负载：seqlen=4680（也即qLen=4680），chunk size=3，latents共7个chunk。

可以采用ncu profile范围内的所有kernel：

```bash
ncu --nvtx --set full -o ${target} python examples/inference/profiling/profile_single_transformer_block.py
```

之前为了测试duration，我封装了一个simple_profiler，所以如果在代码里看到
```python
get_current_simple_profiler().enter("self_attn")
```
类似上面的代码，可以认为等价于：

```python
torch.cuda.nvtx.range_push("self_attn")
```

所以也可以使用`--nvtx-include`

```bash
ncu --nvtx --set full --nvtx-include "self_attn/" -o ${target} python examples/inference/profiling/profile_single_transformer_block.py
```

上面这条命令可以用来profile `self_attn`这个插桩范围内的kernel。

除此以外，我还写了另外两个文件来profile单个DiT block。

`examples/inference/profiling/profile_single_ts_block_args.py`这个文件会设定若干可调的负载参数，从而测试不同的seqlen，chunk size等情况下的运行结果。

但这个文件profile的是所有chunk的情况，我写了另一个文件`examples/inference/profiling/profile_single_ts_block_args.py_with_chunk_idx.py`，额外指定具体跑的是哪个chunk idx的profile。

如果要测试的话，可以先用固定负载的那个文件进行测试

### VAE

我抽象了一个runtime，放在`examples/inference/profiling/profile_wan_vae_decoder.py`中，可以直接运行执行profile，latent的形状是和seqlen=4680的dit profile对齐的

可以类似地执行下面的命令进行profile：

```bash
ncu --nvtx --set full --nvtx-include "WanDecoder_only/" -o ${target} python examples/inference/profiling/profile_wan_vae_decoder.py
``

