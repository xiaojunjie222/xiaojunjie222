<section className="mb-2">
        <h2 className="text-2xl font-semibold text-blue-700">
          自变量机器人 | Harrix 具身推理服务
        </h2>
        <hr className="border-t-2 border-blue-500 my-1" />
        <div>
          <h4 className="font-bold text-blue-800">
            极致交付奖 · Policy Serving · PI0.5 · Token Billing
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              【业务背景】：Harrix 是机器人策略推理引擎。多客户端走 WebSocket 打到 GPU，要动态 batch、多模型族、JAX/Torch 双引擎，还要把 token 用量计进计费，且上报不能拖慢推理。
            </li>
            <li className="list-[circle] list-item">
              【职责&成果】：参与 Harrix serving：通用 serving 边界、PI0.5 流水线、RLT 路径，以及按请求计量 token 并异步上报计费。
            </li>
          </ul>
          <h4 className="font-bold">
            通用 Serving 框架
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              WebSocket Policy Server：请求排队、动态 batch、ServingHandler 编解码、ModelExecutor 上 GPU。
            </li>
            <li className="list-[circle] list-item">
              把模型族和引擎边界拆开：PI0.5 family 与 JAX / Torch runtime 分离，请求适配和 executor 流水线解耦。
            </li>
            <li className="list-[circle] list-item">
              统一 robot state / action 语义，收敛 RLT serving 与单实例部署路径。
            </li>
          </ul>
          <h4 className="font-bold">
            PI0.5 推理与多引擎
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              PI0.5 Torch / JAX serving：epilogue proprio、dataset v2、batched forward、RLT actor 路径。
            </li>
            <li className="list-[circle] list-item">
              修推理正确性：视觉投影 bias、本体感觉输出、张量设备对齐，避免 serving 和训练合同步。
            </li>
          </ul>
          <h4 className="font-bold">
            近期改动（只写能力变化，不含内部实现）
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              通用 serving 边界：PI0.5 流水线独立出来，请求适配和 executor 解耦，旧 launcher 收敛掉。
            </li>
            <li className="list-[circle] list-item">
              模型族和引擎拆开：PI0.5 family 与 JAX / Torch runtime 分界，RLT 走各自 runner。
            </li>
            <li className="list-[circle] list-item">
              推理正确性：视觉投影补 bias，epilogue 本体感觉对齐训练侧，batched forward 设备一致。
            </li>
            <li className="list-[circle] list-item">
              Token 计量：每次响应带 input/output/total tokens，多副本只记一次。
            </li>
            <li className="list-[circle] list-item">
              计费上报：HTTP 批量送 ingest，队列非阻塞；ingest 挂了不拖推理延迟。计量默认关，配置打开。
            </li>
          </ul>
        </div>
</section>


## 机器人动作推理优化笔记

[Harrix 机器人推理优化详解](./notes/robot-inference-optimization.md)

结合 Harrix 的动作推理路径，梳理客户端与服务调度、专家专用化、视觉结构复用、prefix KV、去噪循环、CUDA Graph、权重峰值、算子融合和 FlashAttention，并说明 RTC 计划衔接与跨请求历史的区别。

文章沿实际推理链路展开，附 5 张流程图、动作块与去噪轮次对照表、源码阅读入口和性能定位方法。
