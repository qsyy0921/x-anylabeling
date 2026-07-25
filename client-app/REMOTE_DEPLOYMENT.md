# Remote-only client

客户端不保存或下载模型。首次使用时：

1. 在 `client-app` 创建 `.venv` 并安装本目录项目。
2. 把服务器 API Key 写入 `client-app/secrets/api_key`，文件只包含一行。
3. 配置 SSH 别名 `4090`，确保可免密登录服务器。
4. 运行 `powershell -ExecutionPolicy Bypass -File scripts/start-langgao-client.ps1`。

启动器会建立 `18618` SSH 隧道，将模型目录替换为唯一的
`Remote-Server` 入口，并从服务器动态读取可用模型。

只有管理员需要 `client-app/secrets/model_upload_key`。普通用户没有这个
文件时仍可标注和读取授权数据，但不能上传模型。
