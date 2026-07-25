# Langgao GPU server

运行目录默认是 `/data/mfl/autolabel`：

- 服务源码：`/data/mfl/autolabel/server`
- 模型权重：`/data/mfl/autolabel/models`
- API Key：`/data/mfl/autolabel/secrets/api_key`
- 模型上传管理员密钥：`/data/mfl/autolabel/secrets/model_upload_key`
- 监听地址：`127.0.0.1:18618`

启动：

```bash
AUTOLABEL_ROOT=/data/mfl/autolabel \
  ./scripts/start-langgao-server.sh
```

服务只监听回环地址，客户端必须通过 SSH 隧道访问。模型权重、密钥、
数据和日志均不得提交到 Git。
