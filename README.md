<div align="center">
  <img src="https://readme-typing-svg.herokuapp.com?font=Cascadia+Code&weight=700&size=17&color=0078D7&center=true&pause=1000&width=600&lines=Hi,+I'm+Xiao+Junjie.;Agent+%26+Inference+Engineer;Tool-Use+·+Eval+·+SFT+%2F+Distill" alt="Hi"/>
  <p>
    <img src="https://raw.githubusercontent.com/MartinHeinz/MartinHeinz/master/wave.gif" width="28px" height="28px"/>
    <strong>I'm currently Focus on Agent Tool-Use, Eval, SFT/Distill and Inference.</strong>
  </p>
  <p>
    <code>爱喝鸭屎香柠檬茶</code>
  </p>
</div>

<table>
<tr>
<td valign="top" width="50%">

#### 💼 Work Experience

> 📄 [View Experience →](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/Awards.md)

<p>
<strong>自变量机器人</strong> -- <code>Harrix Serving · PI0.5 · Token Billing</code><br/>
<strong>科大讯飞</strong> -- <code>Agent Tool-Use · Dialogue · Eval · SFT/Distill</code><br/>
<strong>华为</strong> -- <code>MetaERP · 供应链 · 应付发票 · 断供替换</code><br/>
</p>

> 🏆 [View Awards →](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/Awards.md)

- 🏅 **华为 · 英雄强渡大渡河奖** — [MetaERP 替换](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/huawei.md)
- 🏅 **科大讯飞 CBG 首届黑客马拉松优秀奖** — [搭了个搭](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/iflytek.md)
- 🏅 **自变量机器人 · 极致优化奖** — [Harrix Serving](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/x2robot.md)

</td>
<td valign="top" width="48%">

#### 💻 Open Source Experience

> 📂 [View Projects →](https://github.com/xiaojunjie222/xiaojunjie222/blob/main/Awards.md)

- 🧠 **[llm-from-scratch](https://github.com/xiaojunjie222/llm-from-scratch)** — 从零手写 Transformer，支持思维链推理
- ⚙️ **[llm-quantization-toolkit](https://github.com/xiaojunjie222/llm-quantization-toolkit)** — Qwen / DeepSeek / LLaMA 量化
- 🔐 **[rbac-permission-system-java](https://github.com/xiaojunjie222/rbac-permission-system-java)** — Spring Boot + JWT 权限系统
- ⏱️ **[time-dashboard](https://github.com/xiaojunjie222/time-dashboard)** — 时间管理 Web 应用

</td>
</tr>
</table>

---

#### 📝 具身模型与推理笔记

- **[Wall-X 算子优化：融合、访存与数据布局](./notes/wallx-operator-optimization.md)**
  整理 SwiGLU、RMSNorm、RoPE、QKV 后处理、固定双专家重排、原生 GQA 与 GEMM 对齐，逐项说明优化机制、收益来源、适用条件和验证方法。

- **[Harrix 机器人推理优化详解](./notes/robot-inference-optimization.md)**
  从客户端到动作执行，梳理 prefill、KV cache、denoise、算子融合与 CUDA Graph，附 5 张流程图、张量变化表及性能定位方法。

- **[VGA 推理加速](./notes/vga-inference-acceleration.md)**
  VGA 是世界动作模型：先在 latent 里生成未来画面，再从画面特征里读出动作。笔记对照 native 引擎源码，说明每一处已经落地的加速省掉的是哪一段时间。

- **[机器人动作 Tokenizer 与多模态训练详解](./notes/action-tokenizer-and-multimodal-training.md)**
  从动作坐标与 6D 旋转表示出发，拆解 Encoder、RVQ 码本、Decoder 和两阶段训练，再串起图像、指令、动作 token 的拼接、AR 监督与评测。

#### 🔜 计划落地

- **双臂机器人操作**
  针对 pick and place、叠衣服等场景单独训练，训练完成后再单独做推理优化。参考 [SwanLab · LeRobot 具身智能入门](https://docs.swanlab.cn/examples/robot/lerobot-guide.html)。这一项还没落地。

#### 📌 Experience Details

- [自变量机器人 | Harrix 具身推理服务](./x2robot.md)
- [科大讯飞 | Agent 工具调用 · 对话 · 评测 · 微调蒸馏](./iflytek.md)
- [华为 | S 级 MetaERP 替换 · 供应链 / 应付发票](./huawei.md)

<div align="center">
  <img src="https://skillicons.dev/icons?i=python,go,java,linux,docker,pytorch,fastapi,spring" alt="skills"/>
  <br/>
  <img src="https://github-readme-stats.vercel.app/api?username=xiaojunjie222&show_icons=true&theme=transparent&hide_border=true" alt="GitHub stats"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=xiaojunjie222&layout=compact&theme=transparent&hide_border=true" alt="Top langs"/>
</div>
