//! ToolRegistry：按名注册 / 查找工具（spec #1 交付要求）。

use std::collections::HashMap;

use async_trait::async_trait;

use super::{Tool, ToolContext, ToolExecutor, ToolOutput};

/// 注册表中的条目：工具声明（暴露给 LLM 的元数据）+ 执行器。
pub struct RegistryEntry {
    pub tool: super::Tool,
    pub executor: std::sync::Arc<dyn ToolExecutor>,
}

/// 按名注册 / 查找的工具注册表。
#[derive(Default)]
pub struct ToolRegistry {
    entries: HashMap<String, RegistryEntry>,
}

#[derive(Debug, thiserror::Error)]
pub enum ToolRegistryError {
    /// 未知名。消息列出已注册的工具名，帮助模型 / 调用方纠正。
    #[error("unknown tool `{name}`; available tools: {available}")]
    UnknownTool { name: String, available: String },
    /// 重名注册。
    #[error("tool `{name}` is already registered")]
    DuplicateTool { name: String },
    /// 执行失败（未知名或执行器返回错误变体），供 agent loop 一次性调用。
    #[error("{0}")]
    Execute(String),
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注册一个工具：声明 + 执行器。重名返回 [`ToolRegistryError::DuplicateTool`]。
    pub fn register(
        &mut self,
        tool: super::Tool,
        executor: std::sync::Arc<dyn ToolExecutor>,
    ) -> Result<(), ToolRegistryError> {
        if self.entries.contains_key(&tool.name) {
            return Err(ToolRegistryError::DuplicateTool { name: tool.name });
        }
        self.entries.insert(
            tool.name.clone(),
            RegistryEntry { tool, executor },
        );
        Ok(())
    }

    /// 按名查找工具声明。
    pub fn get(&self, name: &str) -> Result<&super::Tool, ToolRegistryError> {
        self.entries
            .get(name)
            .map(|e| &e.tool)
            .ok_or_else(|| self.unknown_tool_err(name))
    }

    /// 已注册的全部工具声明（按名排序，输出稳定，便于生成 tools 声明与测试断言）。
    pub fn tools(&self) -> Vec<super::Tool> {
        let mut tools: Vec<super::Tool> = self
            .entries
            .values()
            .map(|e| e.tool.clone())
            .collect();
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        tools
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// 端到端入口（spec #1 交付行为）：给一个 Tool Call（名 + JSON 参数），
    /// 查找 → 校验 → 执行，返回三态可区分的结果。
    pub async fn execute(
        &self,
        name: &str,
        args: serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<ToolOutput, ToolRegistryError> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| self.unknown_tool_err(name))?;
        if let Err(validation) = super::validate_args(name, &entry.tool.parameters, &args) {
            return Ok(ToolOutput::ValidationFailed(validation.to_string()));
        }
        Ok(entry.executor.execute(args, ctx).await)
    }

    fn unknown_tool_err(&self, name: &str) -> ToolRegistryError {
        let mut names: Vec<&str> = self.entries.keys().map(|k| k.as_str()).collect();
        names.sort_unstable();
        ToolRegistryError::UnknownTool {
            name: name.to_string(),
            available: if names.is_empty() {
                "(none)".to_string()
            } else {
                names.join(", ")
            },
        }
    }
}

/// 注册表的执行入口本质上就是 ToolExecutor（方便把整个注册表当单个工具集使用）。
#[async_trait]
impl ToolExecutor for ToolRegistry {
    fn name(&self) -> &str {
        "registry"
    }

    async fn execute(&self, _args: serde_json::Value, _ctx: &ToolContext) -> ToolOutput {
        ToolOutput::ExecutionError(
            "execute a tool by calling `ToolRegistry::execute(name, args, ctx)` with a tool name, not the registry itself".to_string(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::{ReadTool, Tool, WriteTool};
    use serde_json::json;

    fn dummy_tool(name: &str) -> Tool {
        Tool {
            name: name.into(),
            description: "dummy".into(),
            parameters: json!({"type": "object"}),
        }
    }

    struct NoopExecutor {
        tool_name: String,
    }

    #[async_trait]
    impl ToolExecutor for NoopExecutor {
        fn name(&self) -> &str {
            &self.tool_name
        }
        async fn execute(&self, _args: serde_json::Value, _ctx: &ToolContext) -> ToolOutput {
            ToolOutput::Success("noop".into())
        }
    }

    fn sample_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        reg.register(dummy_tool("alpha"), std::sync::Arc::new(NoopExecutor { tool_name: "alpha".into() }))
            .unwrap();
        reg.register(dummy_tool("beta"), std::sync::Arc::new(NoopExecutor { tool_name: "beta".into() }))
            .unwrap();
        reg
    }

    #[test]
    fn register_and_lookup_by_name() {
        let reg = sample_registry();
        assert_eq!(reg.len(), 2);
        let t = reg.get("alpha").unwrap();
        assert_eq!(t.name, "alpha");
        assert_eq!(reg.tools().len(), 2);
    }

    #[test]
    fn unknown_name_returns_distinguishable_error() {
        let reg = sample_registry();
        let err = reg.get("nope").unwrap_err();
        assert!(matches!(err, ToolRegistryError::UnknownTool { .. }));
        let msg = err.to_string();
        assert!(msg.contains("nope"), "{msg}");
        assert!(msg.contains("alpha") && msg.contains("beta"), "should list available: {msg}");
    }

    #[test]
    fn duplicate_registration_rejected() {
        let mut reg = sample_registry();
        let err = reg
            .register(dummy_tool("alpha"), std::sync::Arc::new(NoopExecutor { tool_name: "alpha".into() }))
            .unwrap_err();
        assert!(matches!(err, ToolRegistryError::DuplicateTool { .. }));
    }

    #[test]
    fn tools_sorted_stably() {
        let reg = sample_registry();
        let names: Vec<String> = reg.tools().into_iter().map(|t| t.name).collect();
        assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);
    }

    #[tokio::test]
    async fn execute_dispatches_to_registered_executor() {
        let reg = sample_registry();
        let out = reg
            .execute("alpha", json!({}), &ToolContext::new("."))
            .await
            .unwrap();
        assert_eq!(out, ToolOutput::Success("noop".into()));
    }

    #[tokio::test]
    async fn execute_unknown_name_is_error() {
        let reg = sample_registry();
        let err = reg
            .execute("missing", json!({}), &ToolContext::new("."))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolRegistryError::UnknownTool { .. }));
    }

    #[tokio::test]
    async fn execute_invalid_args_returns_validation_failed_not_execution_error() {
        // 校验失败走 ToolOutput::ValidationFailed，与执行错误可区分
        let dir = tempfile::tempdir().unwrap();
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool {
                name: "read".into(),
                description: "read".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"]
                }),
            },
            std::sync::Arc::new(ReadTool),
        )
        .unwrap();
        let out = reg
            .execute("read", json!({}), &ToolContext::new(dir.path()))
            .await
            .unwrap();
        match out {
            ToolOutput::ValidationFailed(msg) => assert!(msg.contains("file_path"), "{msg}"),
            other => panic!("expected ValidationFailed, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn end_to_end_write_then_read_via_registry() {
        let dir = tempfile::tempdir().unwrap();
        let ctx = ToolContext::new(dir.path());
        let mut reg = ToolRegistry::new();
        reg.register(
            Tool {
                name: "write".into(),
                description: "write".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["file_path", "content"]
                }),
            },
            std::sync::Arc::new(WriteTool),
        )
        .unwrap();
        reg.register(
            Tool {
                name: "read".into(),
                description: "read".into(),
                parameters: json!({
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"]
                }),
            },
            std::sync::Arc::new(ReadTool),
        )
        .unwrap();

        let out = reg
            .execute("write", json!({"file_path": "nested/a.txt", "content": "hello"}), &ctx)
            .await
            .unwrap();
        assert!(out.is_success(), "{out:?}");

        let out = reg
            .execute("read", json!({"file_path": "nested/a.txt"}), &ctx)
            .await
            .unwrap();
        assert_eq!(out.success_text(), Some("hello"));
    }
}
