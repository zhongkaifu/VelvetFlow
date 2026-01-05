# 安全保存与使用 OpenAI Endpoint Token 的建议

为了运行需要真实 OpenAI 调用的集成测试（例如 `tests/test_openai_live.py`），需要正确地配置并保护 OpenAI 的访问 Token。以下指引旨在帮助你安全地保存、加载和轮换凭证，避免泄露风险。

## 1. 在本地安全存储 Token
- **使用环境变量**：优先将 Token 写入 `OPENAI_API_KEY` 环境变量。临时使用时可以在当前 shell 中执行：
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
- **使用 .env 文件但避免提交**：可以在项目根目录创建 `.env` 文件（或使用 `direnv`/`dotenv` 等工具自动加载），并在其中写入 `OPENAI_API_KEY`。确保 `.env` 已加入 `.gitignore`，避免被提交到仓库。
- **锁定访问权限**：为保存 Token 的文件设置严格权限，例如 `chmod 600 .env`，防止同机其他用户读取。

## 2. 在 CI/CD 或服务器上管理密钥
- **使用机密管理服务**：优先将 Token 存放在 CI 的 Secret 管理（如 GitHub Actions Secrets、GitLab CI Variables），在流水线运行时以环境变量方式注入。
- **分角色密钥**：为 CI、开发机、生产环境分别使用不同 Token，限制误用范围。
- **最小权限与可审计性**：若使用可配置的服务账号或资源组，限制 Token 仅可访问所需的模型/区域，并开启访问日志。

### GitHub Actions 专用指引
- CI workflow 会在检测到 `OPENAI_API_KEY` 已通过 GitHub Secrets 注入时自动运行 `tests/test_openai_live.py`，否则跳过该 Job。
- 建议在仓库或环境级 Secret 中配置以下键值（缺失则为空字符串，Job 会被跳过）：
  - `OPENAI_API_KEY`（必需，存在时才会触发）
  - `OPENAI_BASE_URL`（可选）
  - `OPENAI_MODEL` 与 `OPENAI_EMBEDDING_MODEL`（可选，覆盖默认模型）

## 3. 运行真实 OpenAI 调用的测试
- 本仓库的 `tests/test_openai_live.py` 会在检测到缺失 `OPENAI_API_KEY` 时自动 `skip`，避免在无凭证或离线环境下失败。
- 如需指向非默认的端点，可设置 `OPENAI_BASE_URL`；如需覆盖默认模型，可设置 `OPENAI_MODEL`（聊天）和 `OPENAI_EMBEDDING_MODEL`（向量）。
- 建议在本地或 CI 中仅在必要时运行该文件，例如：
  ```bash
  OPENAI_API_KEY=sk-... pytest -q tests/test_openai_live.py
  ```

## 4. 防止泄露的额外措施
- **永不写死 Token**：不要将 Token 写入源码、配置文件或示例代码；在交付文档或日志前，使用搜索工具确认未包含 `sk-` 等敏感模式。
- **避免分享控制台输出**：在演示或调试时，避免在终端打印完整的请求或错误信息中暴露 Token。
- **定期轮换**：设置定期轮换计划，更新 CI Secrets 和本地 `.env`，并验证旧 Token 已失效。
- **撤销已泄露的 Token**：一旦怀疑泄露，立即在 OpenAI 控制台撤销，并替换相关环境中的 Token。

通过以上措施，可以在保障安全的前提下运行需要真实 OpenAI 访问的测试，并确保凭证在本地与云环境中都得到妥善保护。
