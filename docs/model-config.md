# 模型接入配置说明

测试和运行 pi-rs 所需的 LLM API 配置（endpoint、api key、模型 id）来自本机已有的 Deepexi 网关配置：

- 来源文件：`C:\Users\WX847\.codebuddy\models.json`（本机 CodeBuddy 配置，**不复制进仓库**，避免密钥入库）
- 所有模型均走 OpenAI 兼容协议（`/v1/chat/completions`），与本项目的单 provider 选型一致
- 测试时常用：`glm-5.3-flash`（快、便宜、支持 tool call）

在项目中使用方式：运行时通过环境变量或本机路径读取该配置，`.gitignore` 中确保不落任何 key。
