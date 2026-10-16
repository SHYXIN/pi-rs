//! Tool 系统（GLOSSARY.md：Tool / Tool Call）。
//!
//! 本模块包含四部分：
//! - `Tool` 值类型（名称 / 描述 / 参数 JSON Schema）
//! - `ToolExecutor` 异步执行接口 + `ToolOutput` 三态结果
//! - `ToolRegistry` 注册表（按名注册 / 查找）
//! - 参数 JSON Schema 轻量校验（`validate_args`）
//!
//! 第一版边界内的五个工具（read/bash/edit/write/grep）已全部实现并
//! 接入注册表；schema 声明与实现同文件演进（各工具的 `declaration()`），
//! 一步式构建入口是 `registry::default_registry()`。

mod bash;
mod edit;
mod grep;
mod read;
mod registry;
mod validate;
mod write;

pub use bash::BashTool;
pub use edit::EditTool;
pub use grep::GrepTool;
pub use read::ReadTool;
pub use registry::{default_registry, ToolRegistry, ToolRegistryError};
pub use validate::{validate_args, ValidationError};
pub use write::WriteTool;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// 暴露给 LLM 的一个可调用能力。
///
/// parameters 用 `serde_json::Value` 直接存 JSON Schema 对象。
/// 选型说明：不引 schemars——本项目的工具集是手写固定的（第一版：read/bash/edit/write/grep），
/// schema 由人维护，schemars 的「Rust 类型推导 schema」能力用不上，反而多一层依赖与
/// 生成行为的黑盒；参数校验基于 `Value` 写轻量校验（见 [`validate_args`]）。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    /// 唯一标识
    pub name: String,
    /// 给模型看的功能说明，模型据此决定是否调用
    pub description: String,
    /// 参数 schema（JSON Schema 对象，如 {"type":"object","properties":{...}}）
    pub parameters: serde_json::Value,
}

impl Tool {
    /// 轻量校验：schema 结构本身合法性与完整校验是后续 ticket 的事，
    /// 这里只保证 parameters 是 JSON object（OpenAI function calling 的最低要求）。
    pub fn is_well_formed(&self) -> bool {
        self.parameters.is_object()
    }
}

/// 一次工具执行的结构化结果，三态可区分（spec #1 交付要求）：
/// 成功输出 / 参数校验失败 / 执行错误。
///
/// 错误变体的消息措辞要求「可直接回传给模型重试」：说清哪里错了、
/// 期望什么格式，不含内部细节（路径回显、io 链）留给执行错误变体。
#[derive(Debug, Clone, PartialEq)]
pub enum ToolOutput {
    /// 工具成功执行，文本输出将作为 tool result 回传 LLM
    Success(String),
    /// 参数不符合 schema；message 已带字段名与期望类型，可原样回传模型重试
    ValidationFailed(String),
    /// 校验通过但执行失败（IO 错误、路径非法等）；message 可回传模型
    ExecutionError(String),
}

impl ToolOutput {
    /// 是否成功。
    pub fn is_success(&self) -> bool {
        matches!(self, ToolOutput::Success(_))
    }

    /// 成功时的文本输出；失败变体返回 None。
    pub fn success_text(&self) -> Option<&str> {
        match self {
            ToolOutput::Success(s) => Some(s),
            _ => None,
        }
    }
}

/// ToolOutput → 回传 LLM 的 ToolResult 消息（agent loop 与 REPL 共用的
/// 单一映射，hardening 修复：此前 agent.rs / repl.rs 各抄一份 4 臂 match）。
///
/// 校验失败 / 执行失败 / 注册表级错误（未知工具名等）一律转成
/// is_error=true 的 ToolResult 回传模型（模型可自行纠正重试），
/// 循环不因此中断。
pub fn into_tool_result(
    output: Result<ToolOutput, ToolRegistryError>,
    call_id: &str,
) -> crate::transcript::Message {
    let (content, is_error) = match output {
        Ok(ToolOutput::Success(text)) => (text, false),
        Ok(ToolOutput::ValidationFailed(msg)) => (msg, true),
        Ok(ToolOutput::ExecutionError(msg)) => (msg, true),
        Err(e) => (e.to_string(), true),
    };
    crate::transcript::Message::ToolResult {
        call_id: call_id.to_string(),
        content,
        is_error,
    }
}

/// 工具执行上下文。
///
/// 第一版只携带工作目录：read/write 的相对路径以此为基准解析。
/// 后续 ticket（bash 等）需要时再扩展字段，不预设抽象。
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// 工具执行的工作目录；相对路径的解析基准
    pub working_dir: std::path::PathBuf,
}

impl ToolContext {
    pub fn new(working_dir: impl Into<std::path::PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }
}

/// 工具执行接口（async，tokio 环境，见 ADR-0001）。
///
/// 用 `#[async_trait]` 生成 `Box<dyn Future>` 签名，注册表才能以
/// `Box<dyn ToolExecutor>` 持有异构工具。实现方不做参数校验——
/// 注册表统一先 `validate_args` 再调 `execute`，实现可假设 args 已过校验。
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// 工具名，与注册键一致（用于错误消息回显）。
    fn name(&self) -> &str;

    /// 执行工具。args 为已通过 schema 校验的 JSON object。
    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_roundtrip() {
        let t = Tool {
            name: "read".into(),
            description: "Read a file from disk".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        };
        let json = serde_json::to_string(&t).unwrap();
        let back: Tool = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn well_formed_accepts_object_schema() {
        let t = Tool {
            name: "read".into(),
            description: "Read a file from disk".into(),
            parameters: serde_json::json!({"type": "object"}),
        };
        assert!(t.is_well_formed());
    }

    #[test]
    fn well_formed_rejects_non_object_schema() {
        let t = Tool {
            name: "read".into(),
            description: "Read a file from disk".into(),
            parameters: serde_json::json!("not-a-schema"),
        };
        assert!(!t.is_well_formed());
    }

    #[test]
    fn output_three_states_are_distinguishable() {
        let ok = ToolOutput::Success("done".into());
        let bad = ToolOutput::ValidationFailed("missing field".into());
        let err = ToolOutput::ExecutionError("io failed".into());
        assert!(ok.is_success());
        assert!(!bad.is_success());
        assert!(!err.is_success());
        assert_eq!(ok.success_text(), Some("done"));
        assert_eq!(bad.success_text(), None);
        assert_ne!(
            std::mem::discriminant(&ok),
            std::mem::discriminant(&bad)
        );
        assert_ne!(
            std::mem::discriminant(&bad),
            std::mem::discriminant(&err)
        );
    }
}
