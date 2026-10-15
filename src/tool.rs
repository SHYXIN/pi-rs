//! Tool 值类型（GLOSSARY.md：暴露给 LLM 的一个可调用能力）。

use serde::{Deserialize, Serialize};

/// 暴露给 LLM 的一个可调用能力。
///
/// parameters 用 `serde_json::Value` 直接存 JSON Schema 对象。
/// 选型说明：不引 schemars——本项目的工具集是手写固定的（第一版：read/bash/edit/write/grep），
/// schema 由人维护，schemars 的「Rust 类型推导 schema」能力用不上，反而多一层依赖与
/// 生成行为的黑盒；工具参数校验逻辑本身也是后续 ticket 的事，届时基于 `Value` 写
/// 轻量校验即可。
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Tool {
        Tool {
            name: "read".into(),
            description: "Read a file from disk".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "path": {"type": "string"}
                },
                "required": ["path"]
            }),
        }
    }

    #[test]
    fn tool_roundtrip() {
        let t = sample();
        let json = serde_json::to_string(&t).unwrap();
        let back: Tool = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
    }

    #[test]
    fn well_formed_accepts_object_schema() {
        assert!(sample().is_well_formed());
    }

    #[test]
    fn well_formed_rejects_non_object_schema() {
        let t = Tool {
            parameters: serde_json::json!("not-a-schema"),
            ..sample()
        };
        assert!(!t.is_well_formed());
    }
}
