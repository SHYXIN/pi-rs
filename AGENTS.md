# AGENTS.md

pi-rs：用 Rust 实现的 coding agent，以 pi（TypeScript 版）为参照系。

## Agent skills

### Issue tracker

Issues live in GitHub Issues of SHYXIN/pi-rs (use `gh` CLI). See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`), recorded as a `Status:` line in each issue file. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: `GLOSSARY.md` at repo root, ADRs in `docs/adr/`. See `docs/agents/domain.md`.

## 工程约定

- 异步运行时统一 tokio（见 ADR-0001），不引入其他运行时
- 工具选择完全依赖 LLM 原生 function calling，不做关键词路由
- 密钥不入库：模型配置来自本机 `~/.codebuddy/models.json`（见 `docs/model-config.md`），`.gitignore` 挡 `.env`
- 领域术语以 `GLOSSARY.md` 为准（Agent Loop / Tool / Tool Call / Provider / Transcript / First Version Boundary / Provider Parity Principle）
