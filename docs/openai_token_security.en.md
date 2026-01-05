## OpenAI Token Security (English)

> 中文版本: [openai_token_security.md](openai_token_security.md)

- **Secure storage**: Keep `OPENAI_API_KEY` in environment variables or a secrets manager; avoid committing keys to source control.
- **Integration tests**: Use explicit toggles to enable tests that call OpenAI APIs; default to offline/simulated runs to prevent accidental usage.
- **Least privilege**: Prefer project-scoped keys with usage limits; rotate credentials regularly and monitor for unexpected activity.
