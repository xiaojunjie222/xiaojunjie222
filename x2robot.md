<section className="mb-2">
        <h2 className="text-2xl font-semibold text-blue-700">
          自变量机器人 | AI 推理服务
        </h2>
        <hr className="border-t-2 border-blue-500 my-1" />
        <div>
          <h4 className="font-bold text-blue-800">
            工厂离线推理 License · mTLS 代理 · 到期管控
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              【业务背景】：机器人进客户工厂后经常离线。推理服务要能在现场跑，同时管住授权：谁能连、连几台、过期停服，还不能改业务客户端和推理进程本身。
            </li>
            <li className="list-[circle] list-item">
              【职责&成果】：从 0 落地离线推理授权与安全通信：机器人到推理机的 mTLS 透明代理、License Daemon 校验、普通 U 盘离线导入、本机状态页，以及 TLS / 过期 / 大消息流式转发的端到端测试。
            </li>
          </ul>
          <h4 className="font-bold">
            推理链路透明代理
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              机器人连本机 Client Proxy，推理机 Server Proxy 用 TLS 1.3 双向证书建连，再转到本机推理服务。
            </li>
            <li className="list-[circle] list-item">
              代理只做 WebSocket 透明转发：不改业务协议、不解析应用消息、不额外加应用层认证，业务客户端和推理服务都不用改。
            </li>
          </ul>
          <h4 className="font-bold">
            License 授权与到期关闭
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              Daemon 校验签名、机器绑定、离线可信时间、严格递增序号、机器人证书身份和并发连接数。
            </li>
            <li className="list-[circle] list-item">
              License 失效后拒绝新连接、停止转发新的上行，已发出请求排空后关闭，避免过期后继续白嫖推理。
            </li>
          </ul>
          <h4 className="font-bold">
            离线交付与本机状态
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              现场用普通 U 盘导入 License：只读扫描、校验失败保留当前有效授权，通过后原子更新。
            </li>
            <li className="list-[circle] list-item">
              本机状态页只读展示授权是否有效，并可试连代理；ACTIVE 可通，EXPIRED 拒绝新连接。
            </li>
          </ul>
          <h4 className="font-bold">
            测试与交付
          </h4>
          <ul className="ml-4">
            <li className="list-[circle] list-item">
              Go 端到端覆盖真实 TLS 1.3、双向证书、metadata 首帧、帧边界、大消息流式转发、异常关闭和过期拒绝。
            </li>
            <li className="list-[circle] list-item">
              兼容真实 RobotClient 走代理；linux/amd64、linux/arm64 构建，配套 systemd / udev 现场部署。
            </li>
          </ul>
        </div>
</section>
