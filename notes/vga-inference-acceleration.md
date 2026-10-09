# VGA 推理加速

这份说明只写 VGA native 引擎里已经落地的加速，以及对应的源码在做什么。

范围是 `python/harrix/model_executors/vga/engines/native/`。服务入口是 `handlers/vga.py` 的 `create_policy`，引擎名是 `native`。固定 33 帧配置走 `video_rcm_trigflow_action_ode`：学生视频 4 步，动作读教师视频的固定锚点，服务侧不把视频还原成像素。

下面这些名字在讨论里出现过，但这条路径里没有单独的实现：小分辨率 VAE 专用快路径、删掉注意力重排、端侧专用算子、单独的“清理重复搬运”过程。`tiled_encode_batch` 仍把每块结果拷回 CPU；`attention.py` 和 `ops.py` 里的 `rearrange` 还在。

## 总览

| 代码里的做法 | 主要位置 |
|---|---|
| CUDA Graph，录下固定形状的 GPU 操作再重放 | `core/cuda_graph.py`，`sampling/cuda_graph.py`，T5 与 VAE encode |
| 时间条件缓存 | `core/cache.py` 的 `TimeConditionCache` |
| 文字编码和首帧 VAE 两条 CUDA Stream | `sampling/video_inputs.py` |
| 多路相机先叠成一批，再做一次 CPU 到 GPU 拷贝 | `video_inputs.py` 的 `_tensor_frames` |
| 视频层特征留在 GPU | `ExactCache`，`VideoStream.step` 的 `keep_on_device` |
| ODE 路径在动作不再需要视频时提前停 | `streams.py` 的 `early_stop_for_action` |
| 服务不算回视频像素 | `executor.py` 的 `decode_video=False` |
| `rmsnorm`、`rope` 优先走 xcompute | `core/ops.py` |
| 注意力按已安装的库选择实现 | `ops.py` 的 `flash_attention` |

学生视频和教师视频是两套权重。推理代码负责加载，并按 deployment 使用它们。

## CUDA Graph

`ExactCudaGraph` 包住一个只吃张量的函数。第一次用某个输入签名时，它先在另一条 stream 上预热，再用 `torch.cuda.CUDAGraph` 录制。录完立刻重放一次，和录制前的 eager 结果做逐元素比较。对不上就丢掉这张图，并记下 `precision guard failed`。之后同签名的调用只把新输入拷进录制时的静态张量，再 `replay`。

去噪主循环的图在 `Fixed33CudaGraphRunner`。名字是 `VGA 33-frame denoise core`，最多留 1 条。`DenoiseLoop._cuda_graph_eligible` 要求同时满足：配置打开 `enable_cuda_graph`、设备是 CUDA、33 帧、192×224、3 个视角、batch 为 1、动作长度 64、视频模式 `rcm_trigflow` 且 4 步、动作模式 `ode` 且 4 步、动作视频耦合是 `renoise_x0_to_fixed_ode_anchor`、不解码视频、没有 action history、有首帧。少一项就调用 `_execute_rcm`，公式不变。

另外两处独立的图：

- `WanPrompter.encode_prompt` 在 `use_cuda_graph=True` 时用 `ExactCudaGraph` 包住 T5。
- `WanVideoVAE.encode` 的 `_graph_encode` 用 `ExactCudaGraph` 包住 VAE encode，最多 4 条。`single_encode` 和 `tiled_encode_batch` 的每一块都可以走它。

`NativeEngine.reset` 会清掉这三处图，以及时间条件缓存。

## 时间条件缓存

`DenoiseLoop` 构造时总是创建 `TimeConditionCache`，上限 192MB。`_time_cache()` 只在 `deployment.profile_type` 以 `video_rcm_trigflow_action_` 开头时把这个对象交出去。`deployment.py` 里符合的是三份配置：`video_rcm_trigflow_action_ode`、`video_rcm_trigflow_action_restart`、`video_rcm_trigflow_action_consistency`。`teacher_ode` 对不上，调用方拿到 `None`，时间向量每次现算。

视频侧，`model_fn` 进入后先调用 `_cached_time_condition`。缓存是 `None` 就直接跑 `_time_condition`：正弦向量、`time_embedding`、`time_projection`。否则 `get_or_create("video", dit, key, factory)`。key 带上调用方给的时间标记、latent 形状、dtype、设备，以及是否把第一帧钉进 latent。

RCM 学生网络传入的标记是 `("rcm", time, timestep_scale)`。同一次请求的 4 个时间点不同，彼此不会命中；下一次请求形状和时间都不变时才会命中。教师视频走到锚点之前，只有 `prepared` 不是 `None` 才传入缓存，标记是 `("action_video_teacher", 步号, 时间)`。锚点那一次的标记是 `("action_video_anchor", 时间)`。

动作侧在 `ActionStream._predict`。命名空间是 `"action"`，标记是 `("schedule", schedule_key, 第几步)`，再拼上时间和动作张量的 dtype、设备。

`get_or_create` 用 `(namespace, id(model), key)` 查 `OrderedDict`。命中则把条目挪到队尾并返回张量，`factory` 不执行。模型的 `time_embedding` 和 `time_projection` 权重地址或形状变了，该模型名下的旧条目全部丢掉。单条结果超过 192MB 只返回、不保存。

缓存的是时间调制向量。它不保存画面，也不保存动作。

## 文字编码和首帧压缩并行

CUDA 上，`DenoiseLoop` 建两条 stream，放在 `conditioning_streams`。`_prepare_conditioning` 在有首帧图时这样用：

1. 两条 stream 都 `wait_stream` 当前 stream。
2. `text_stream` 上跑 `_prompt_context`，也就是 T5。
3. `vae_stream` 上跑 `_fuse_first_frame`，也就是首帧 VAE encode，并把首帧 latent 写进视频 latent 的第 0 帧。
4. 当前 stream 再等待这两条 stream，并对结果 `record_stream`。

没有 CUDA，或者没有输入图时，这两步在当前 stream 上按顺序做。

## 多路相机合并后再拷到 GPU

`_tensor_frames`：每一帧都在 CPU 上时，先 `torch.stack`，再对这一整批做一次 `.to(device=pipe.device)`。已经在 GPU 上的帧则逐帧拷贝。PIL 路径在 `_pil_frames` 里同样是先堆成一个张量，再一次 `.to`。

三视角时，这一次拷贝带走的是叠好的一批，而不是每路相机各提交一次小传输。

## 视频特征留在 GPU，并直接交给动作

`ExactCache` 在同一次采样里保存这些对象：动作要读的视频 token 和 KV、文字 context 及其 KV、视频 RoPE、视角 RoPE、本体状态嵌入。`model_fn` 看到 `cache.video_rope` 或 `cache.video_context` 已经有值，就不再重算。

ODE 的 `VideoStream.step` 默认把每一层视频特征 `.to("cpu")`。`keep_on_device=True` 时留下 GPU 上的列表。`denoise.py` 的同步路径在取动作要用的那一步时传入 `keep_on_device=True`。不在 `keep_steps` 里的步把特征丢掉，不保留。

动作网络要的是这些层特征，不是像素。`model_fn` 在 `return_video_features=True` 时收集每一层输出。服务路径另外保证不跑 VAE decode：`VGAExecutor.infer_batch` 固定 `decode_video=False`。

## ODE 路径提前停下视频

`VideoStream.num_steps` 在 `early_stop_for_action` 为真、并且存在 `action_mapping` 时，返回映射里最大的视频步再加 1。`DenoiseLoop.run` 把 `early_stop_for_action` 设成 `skip_video_decode`。服务请求因此不会把视频采样跑到 scheduler 的最后一步，只要动作映射已经覆盖到的那一步。

这条逻辑在 `video_mode == "ode"` 的 `VideoStream` 上。固定 33 帧的 `rcm_trigflow` 走 `_run_rcm`，学生网络仍按时间表跑完 4 步，教师网络跑到 deployment 写明的锚点。RCM 路径不使用 `early_stop_for_action`。

## 算子选择

`core/ops.py` 里只有两个算子走可选的外部实现：

```python
rmsnorm = _FallbackOp("rmsnorm", _rmsnorm_pytorch)
rope = _FallbackOp("rope", _rope_pytorch)
```

`_FallbackOp` 第一次调用时 `import xcompute` 并取同名函数。导入失败就用 PyTorch 实现，并打一条 `using PyTorch fallback`。VAE 的 RMSNorm 在调用前把 channel 维挪到最后一维，再展成 `[num_tokens, hidden]`，因为 xcompute 的 rmsnorm 按最后一维归一化，并且要求输入连续。注意力的 RoPE 在 `attention.py` 里把 Q、K 重排成该算子要求的 `(two d)` 布局，算完再排回去。

`flash_attention` 的选择顺序是：有 mask 时用 PyTorch SDPA；否则依次尝试 FlashAttention-3、FlashAttention-2、SageAttention；都没有则再使用 SDPA。每次调用前后仍有 `rearrange`，用来匹配所选库的头维布局。当前代码没有把这些重排删掉。

这个目录不包含 `.cu`。xcompute 不在进程里时，`rmsnorm` 和 `rope` 退回上面的 PyTorch 函数。

## 学生权重和教师权重

`model_stack.py` 在 profile 以 `video_rcm_trigflow_action_` 开头时，除了 `self.dit` 再构建一套 `WanModel` 作为 `_video_teacher`。`load_deployment_state_dict` 分别从 `checkpoints["video_student"]` 和 `checkpoints["video_teacher"]` 加载。`teacher_ode` 只有一套视频权重。

`rcm.py` 的去噪一步用三个由正弦、余弦算出的系数：`c_skip`、`c_out`、`c_in`。网络看 `c_in * x`，输出和当前 `x` 合成更干净的 `x0`，然后把第 0 帧再钉回真实画面。
