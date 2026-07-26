# Langgao X-AnyLabeling

朗高远程 GPU 辅助标注项目。桌面客户端只负责图片浏览、人工标注和
visual prompt 交互；所有模型权重、模型加载和推理均在服务器完成。

## 目录

- `client-app/`：Windows/macOS X-AnyLabeling 客户端源码及远程启动脚本。
- `server-service/`：服务器 GPU 推理 API、模型注册配置和 worker 启动脚本。

## 数据与模型边界

- 模型权重只存放在服务器 `/data/mfl/autolabel/models`。
- 客户端设置 `XANYLABELING_REMOTE_ONLY=1`，禁止下载本地模型。
- 数据模式按路径自动判断：
  - Windows/macOS 本地路径由客户端直接读写；
  - `/data/mfl/langgao/...` 或 `server://...` 路径由服务器读写。
- “Open Server Dataset”只加载服务器文件列表和当前图片，不会把整套图片
  或 JSON 标注同步到客户端；保存、删除标注会直接更新服务器端 JSON。
- 多客户端同时编辑同一张服务器图片时使用标注版本校验；若标注已被其他
  客户端修改，旧客户端会拒绝覆盖并提示重新加载。
- 本地模型只能以 ZIP 包上传到服务器 staging；上传后必须由管理员审核，
  不会自动执行或加入模型列表。
- API Key、SSH 凭据、虚拟环境、图片数据和运行结果禁止提交 Git。
- 客户端通过 SSH 隧道访问服务器回环端口 `127.0.0.1:18618`。

## 当前服务器模型

- EdgeSAM
- GeCo SAM-HQ ViT-H
- GroundingDINO Swin-T + SAM2 Large
- SAM 2.1 Hiera Tiny
- SAM 2.1 Hiera Large
- SAM 3 ViT-H ONNX
- SAM 3.1 Multiplex Official
- SAM-HQ ViT-L
- YOLOv8s + SAM2 Hiera Base

具体权重路径由 `server-service/app/models/legacy_anylabeling.py` 管理。

多用户聊天接入方案见
`docs/MULTI_USER_CHAT_ARCHITECTURE.md`。当前内部反代不能直接分发给客户端。
