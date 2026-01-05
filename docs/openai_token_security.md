# OpenAI 凭证安全指引

本项目会在嵌入检索与工作流规划阶段调用 OpenAI API。为避免泄露凭证，请遵循以下做法：

## 安全存储与加载

- **使用环境变量**：本地开发和 CI 均推荐通过 `OPENAI_API_KEY`、`OPENAI_BASE_URL`（可选）等环境变量传入凭证，不要在代码或配置文件里硬编码。
- **本地 `.env` 文件**：可以将凭证写入未纳入版本控制的 `.env` 文件，并在启动前通过 `export $(cat .env | xargs)` 或使用 `direnv`/`pipenv` 等自动加载。确保 `.env` 文件的权限限制为当前用户（例如 `chmod 600 .env`）。
- **Secret Manager/CI Secrets**：在生产与流水线中，将凭证存储在云密钥管理服务（如 AWS Secrets Manager、GCP Secret Manager）或 CI 平台的 Secret 配置中，通过环境变量下发。
- **避免日志输出**：不要在日志、异常信息或测试基准里打印完整的 token，调试时可只输出前后 4 位或使用占位符。
- **最小化权限**：为自动化测试或演示使用单独的受限 key，限制额度、模型访问范围或 IP 白名单，并在使用完毕后及时轮转。

## 集成测试的安全开关

新增的 `tests/test_openai_workflow_integration.py` 会在以下条件满足时实际调用 OpenAI endpoint 构建工作流：

1. 设置 `VELVETFLOW_RUN_OPENAI_E2E=1` 显式开启。
2. 设置 `OPENAI_API_KEY`（以及可选的 `OPENAI_BASE_URL`/`OPENAI_ORG_ID` 等）用于认证。

示例执行命令：

```bash
VELVETFLOW_RUN_OPENAI_E2E=1 \
OPENAI_API_KEY="sk-..." \
pytest tests/test_openai_workflow_integration.py -q
```

在未设置开关或缺少凭证时，该测试会被自动跳过，避免意外的真实调用。

## 凭证清理

- 在共享机器或 CI 运行后，清除会话中的环境变量：`unset OPENAI_API_KEY OPENAI_BASE_URL OPENAI_ORG_ID`。
- 失效或轮转旧凭证，保持审计记录，及时撤销可疑的访问密钥。
