# VGA 推理加速

VGA 是世界动作模型。当前相机画面和指令进去之后，视频网络先在 latent 里往前生成一段未来画面，动作网络再读这段画面的层特征，走出一段动作。仓库里的名字是 VGA，deployment 的任务名是 `vga_openloop`。

一次推理里有两段预测。未来画面是对接下来世界会怎样的预测。固定 33 帧这条路径上，学生网络用 4 步把画面从噪声里变出来，第 0 帧始终钉在机器人刚看到的画面上。动作网络不看像素，而是读教师视频走到固定锚点时每一层的特征，再走出 64 步动作。

服务把视频像素解码关掉了，回给机器人的只有动作。世界这一段仍然在 latent 里算过，否则动作没有特征可读。deployment 写的是 open-loop：一次调用给出一整段未来画面和一整段动作。机器人要接着做，是外面再发下一次观测，重新走一遍。

下面只写 VGA native 引擎里已经落地的加速：源码在做什么，以及每一块省掉的是哪一段时间。

范围是 `python/harrix/model_executors/vga/engines/native/`。服务入口是 `handlers/vga.py` 的 `create_policy`，引擎名是 `native`。固定 33 帧配置走 `video_rcm_trigflow_action_ode`：学生视频 4 步，动作读教师视频的固定锚点，服务侧不把视频还原成像素。

固定 33 帧的服务请求里，每次都会碰到的是 CUDA Graph、时间条件缓存、两条 Stream、多相机合并拷贝，以及不把视频画成像素。ODE 提前停止只在另一条视频路径上生效。`rmsnorm`、`rope` 和注意力换实现，取决于机器上有没有装对应的库。

下面这些名字在讨论里出现过，但这条路径里没有单独的实现：小分辨率 VAE 专用快路径、删掉注意力重排、端侧专用算子、单独的“清理重复搬运”过程。`tiled_encode_batch` 仍把每块结果拷回 CPU；`attention.py` 和 `ops.py` 里的 `rearrange` 还在。

## 总览

| 代码里的做法 | 省掉的时间 | 主要位置 |
|---|---|---|
| CUDA Graph，录下固定形状的 GPU 操作再重放 | CPU 逐个把算子提交给 GPU | `core/cuda_graph.py`，`sampling/cuda_graph.py`，T5 与 VAE encode |
| 时间条件缓存 | 同一个时间点反复跑两个小网络 | `core/cache.py` 的 `TimeConditionCache` |
| 文字编码和首帧 VAE 两条 CUDA Stream | 两段互不依赖的工作串行相加 | `sampling/video_inputs.py` |
| 多路相机先叠成一批，再做一次 CPU 到 GPU 拷贝 | 每路相机各发起一次小传输 | `video_inputs.py` 的 `_tensor_frames` |
| 视频层特征留在 GPU，服务不还原像素 | 重算不变的向量，以及把 latent 画成图片 | `ExactCache`，`executor.py` 的 `decode_video=False` |
| ODE 路径在动作不再需要视频时提前停 | 动作已经用完之后的视频采样步 | `streams.py` 的 `early_stop_for_action` |
| `rmsnorm`、`rope` 优先走 xcompute | 这两个算子内部的 PyTorch 实现 | `core/ops.py` |
| 注意力按已安装的库选择实现 | 注意力内部的计算和中间矩阵 | `ops.py` 的 `flash_attention` |

学生视频和教师视频是两套权重。推理代码负责加载，并按 deployment 使用它们。固定 33 帧请求里，教师仍要按 50 步时间表走到第 45 步，这一段前向次数比学生那 4 步多。

## CUDA Graph

省的是 CPU 把算子交给 GPU 的时间。算的数学没有变少。

去噪主循环里，同样的 33 帧、192×224、3 个相机、64 步动作，每次请求调用的 GPU 算子顺序是一样的。不做 Graph 时，CPU 要逐个把这些算子提交给 GPU。`ExactCudaGraph` 第一次把整段录下来，并和当场算的结果对一下。对得上，以后同形状的请求只把新输入拷进录好的张量，再重放。

`ExactCudaGraph` 包住一个只吃张量的函数。第一次用某个输入签名时，它先在另一条 stream 上预热，再用 `torch.cuda.CUDAGraph` 录制。录完立刻重放一次，和录制前的 eager 结果做逐元素比较。对不上就丢掉这张图，并记下 `precision guard failed`。之后同签名的调用只把新输入拷进录制时的静态张量，再 `replay`。

去噪主循环的图在 `Fixed33CudaGraphRunner`。名字是 `VGA 33-frame denoise core`，最多留 1 条。`DenoiseLoop._cuda_graph_eligible` 要求同时满足：配置打开 `enable_cuda_graph`、设备是 CUDA、33 帧、192×224、3 个视角、batch 为 1、动作长度 64、视频模式 `rcm_trigflow` 且 4 步、动作模式 `ode` 且 4 步、动作视频耦合是 `renoise_x0_to_fixed_ode_anchor`、不解码视频、没有 action history、有首帧。少一项就调用 `_execute_rcm`，公式不变。

另外两处独立的图：

- `WanPrompter.encode_prompt` 在 `use_cuda_graph=True` 时用 `ExactCudaGraph` 包住 T5。
- `WanVideoVAE.encode` 的 `_graph_encode` 用 `ExactCudaGraph` 包住 VAE encode，最多 4 条。`single_encode` 和 `tiled_encode_batch` 的每一块都可以走它。

第一次请求要花时间录制，从第二次同形状请求才开始省。`NativeEngine.reset` 会清掉这三处图，以及时间条件缓存。

## 时间条件缓存

省的是「这个时间点对应哪组向量」的重复计算。它不省后面那整张大视频网络。

视频网络和动作网络每一步都要知道现在有多吵。这个数会先变成正弦向量，再过两个小网络 `time_embedding` 和 `time_projection`。这两个小网络不看画面内容。时间、形状、模型不变，结果就相同。

`DenoiseLoop` 构造时总是创建 `TimeConditionCache`，上限 192MB。`_time_cache()` 只在 `deployment.profile_type` 以 `video_rcm_trigflow_action_` 开头时把这个对象交出去。`deployment.py` 里符合的是三份配置：`video_rcm_trigflow_action_ode`、`video_rcm_trigflow_action_restart`、`video_rcm_trigflow_action_consistency`。`teacher_ode` 对不上，调用方拿到 `None`，时间向量每次现算。

视频侧，`model_fn` 进入后先调用 `_cached_time_condition`。缓存是 `None` 就直接跑 `_time_condition`。否则 `get_or_create("video", dit, key, factory)`。key 带上调用方给的时间标记、latent 形状、dtype、设备，以及是否把第一帧钉进 latent。

RCM 学生网络传入的标记是 `("rcm", time, timestep_scale)`。同一次请求的 4 个时间点不同，彼此用不上对方的缓存。下一次请求如果还是同样的形状和时间表，才会命中。教师视频走到锚点之前，只有 `prepared` 不是 `None` 才传入缓存，标记是 `("action_video_teacher", 步号, 时间)`。锚点那一次的标记是 `("action_video_anchor", 时间)`。

动作侧在 `ActionStream._predict`。命名空间是 `"action"`，标记是 `("schedule", schedule_key, 第几步)`，再拼上时间和动作张量的 dtype、设备。

`get_or_create` 用 `(namespace, id(model), key)` 查 `OrderedDict`。命中则把条目挪到队尾并返回张量，`factory` 不执行。模型的 `time_embedding` 和 `time_projection` 权重地址或形状变了，该模型名下的旧条目全部丢掉。单条结果超过 192MB 只返回、不保存。

缓存的是时间调制向量。它不保存画面，也不保存动作。

## 文字编码和首帧压缩并行

省的是两段互不依赖的工作串行相加的时间。耗时接近较慢的那一边。

T5 读的是指令，VAE 压的是第一帧相机图。两边算完之前，去噪还不能开始，但它们互相不读对方的结果。CUDA 上，`DenoiseLoop` 建两条 stream，放在 `conditioning_streams`。`_prepare_conditioning` 在有首帧图时这样用：

1. 两条 stream 都 `wait_stream` 当前 stream。
2. `text_stream` 上跑 `_prompt_context`，也就是 T5。
3. `vae_stream` 上跑 `_fuse_first_frame`，也就是首帧 VAE encode，并把首帧 latent 写进视频 latent 的第 0 帧。
4. 当前 stream 再等待这两条 stream，并对结果 `record_stream`。

没有 CUDA，或者没有输入图时，这两步在当前 stream 上按顺序做，这条重叠用不上。

## 多路相机合并后再拷到 GPU

省的是每路相机各发起一次小传输的启动开销。

三路图如果都在 CPU 上，`_tensor_frames` 先 `torch.stack`，再对这一整批做一次 `.to(device=pipe.device)`。一次拷走三路。图已经在 GPU 上时，代码改成逐帧拷，这条合并不用。PIL 路径在 `_pil_frames` 里同样是先堆成一个张量，再一次 `.to`。

## 视频特征留在 GPU，并直接交给动作

省的是两块无用工作：同一次采样里反复重算不变的向量，以及把视频 latent 画成像素。

动作网络要的是视频每一层的特征，不是一张张图片。

- `ExactCache` 在同一次采样里保存动作要读的视频 token 和 KV、文字 context 及其 KV、视频 RoPE、视角 RoPE、本体状态嵌入。`model_fn` 看到 `cache.video_rope` 或 `cache.video_context` 已经有值，就不再重算，也不先搬回 CPU。
- ODE 的 `VideoStream.step` 默认把每一层视频特征 `.to("cpu")`。`keep_on_device=True` 时留下 GPU 上的列表。`denoise.py` 的同步路径在取动作要用的那一步时传入 `keep_on_device=True`。不在 `keep_steps` 里的步把特征丢掉，不保留。
- `model_fn` 在 `return_video_features=True` 时收集每一层输出，直接交给动作。`VGAExecutor.infer_batch` 固定 `decode_video=False`，不跑 VAE decode。服务回给机器人的只有 64 步动作。

## ODE 路径提前停下视频

省的是动作已经用完之后、视频还在继续采样的那些步。这条逻辑只在 `video_mode == "ode"` 上。

`VideoStream.num_steps` 在 `early_stop_for_action` 为真、并且存在 `action_mapping` 时，返回映射里最大的视频步再加 1。`DenoiseLoop.run` 把 `early_stop_for_action` 设成 `skip_video_decode`。服务请求因此停在动作映射已经覆盖到的那一步。

固定 33 帧的 `rcm_trigflow` 走 `_run_rcm`，不使用 `early_stop_for_action`。学生网络仍跑完 4 步。动作用的教师视频按 deployment 里的 50 步时间表走到第 45 步：第 45 步之前每步一次教师网络前向，锚点上再取一次层特征。这段大约 46 次前向，比学生那 4 步多，是这条服务请求里仍然要付的大头。

## 算子选择

省的是单个算子内部的时间。机器上没有对应的库时，速度停在 PyTorch 实现上。

`core/ops.py` 里只有两个算子走可选的外部实现：

```python
rmsnorm = _FallbackOp("rmsnorm", _rmsnorm_pytorch)
rope = _FallbackOp("rope", _rope_pytorch)
```

`_FallbackOp` 第一次调用时 `import xcompute` 并取同名函数。导入失败就用 PyTorch 实现，并打一条 `using PyTorch fallback`。VAE 的 RMSNorm 在调用前把 channel 维挪到最后一维，再展成 `[num_tokens, hidden]`，因为 xcompute 的 rmsnorm 按最后一维归一化，并且要求输入连续。注意力的 RoPE 在 `attention.py` 里把 Q、K 重排成该算子要求的 `(two d)` 布局，算完再排回去。排布局这一步还在。

`flash_attention` 的选择顺序是：有 mask 时用 PyTorch SDPA；否则依次尝试 FlashAttention-3、FlashAttention-2、SageAttention；都没有则再使用 SDPA。选中更快的实现时，省的是注意力内部的计算和中间矩阵。每次调用前后仍有 `rearrange`，用来匹配所选库的头维布局。当前代码没有把这些重排删掉。

这个目录不包含 `.cu`。

## 学生权重和教师权重

这是两套 checkpoint 的用法，不是上面那种运行时开关。

`model_stack.py` 在 profile 以 `video_rcm_trigflow_action_` 开头时，除了 `self.dit` 再构建一套 `WanModel` 作为 `_video_teacher`。`load_deployment_state_dict` 分别从 `checkpoints["video_student"]` 和 `checkpoints["video_teacher"]` 加载。`teacher_ode` 只有一套视频权重。

`rcm.py` 的去噪一步用三个由正弦、余弦算出的系数：`c_skip`、`c_out`、`c_in`。网络看 `c_in * x`，输出和当前 `x` 合成更干净的 `x0`，然后把第 0 帧再钉回真实画面。学生按这个公式走 4 步。动作不读学生的最后一帧，而读教师走到固定锚点时的层特征。
