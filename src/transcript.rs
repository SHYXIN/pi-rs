//! 统一消息类型与 Transcript（GLOSSARY.md：发给 LLM 的上下文的唯一事实来源）。
//!
//! 所有变体 serde 序列化稳定，provider 层据此转成各 API 的请求格式。

use serde::{Deserialize, Serialize};

/// 一次会话中的统一消息。tag = "role" 序列化，保证跨语言/跨版本结构可读。
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "camelCase")]
pub enum Message {
    User {
        content: String,
    },
    Assistant {
        content: String,
    },
    /// LLM 发出的一次工具使用请求（工具名 + 参数）
    ToolCall {
        #[serde(rename = "callId")]
        call_id: String,
        name: String,
        /// 模型生成的参数，JSON object
        arguments: serde_json::Value,
    },
    /// 工具执行结果，回传给 LLM
    ToolResult {
        #[serde(rename = "callId")]
        call_id: String,
        content: String,
        /// 工具执行是否失败（失败时 content 为错误信息）
        is_error: bool,
    },
}

/// 一次会话的完整消息历史容器（GLOSSARY.md：Transcript）。
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Transcript {
    messages: Vec<Message>,
}

impl Transcript {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, message: Message) {
        self.messages.push(message);
    }

    pub fn messages(&self) -> &[Message] {
        &self.messages
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 序列化往返：每个变体经 JSON round-trip 后与原值相等
    #[test]
    fn message_roundtrip_all_variants() {
        let messages = vec![
            Message::User {
                content: "hello".into(),
            },
            Message::Assistant {
                content: "hi there".into(),
            },
            Message::ToolCall {
                call_id: "call_1".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path": "src/main.rs"}),
            },
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "fn main() {}".into(),
                is_error: false,
            },
        ];
        for m in messages {
            let json = serde_json::to_string(&m).unwrap();
            let back: Message = serde_json::from_str(&json).unwrap();
            assert_eq!(back, m);
        }
    }

    #[test]
    fn transcript_roundtrip_preserves_order() {
        let mut t = Transcript::new();
        t.push(Message::User {
            content: "fix the bug".into(),
        });
        t.push(Message::ToolCall {
            call_id: "c1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command": "cargo test"}),
        });
        t.push(Message::ToolResult {
            call_id: "c1".into(),
            content: "test result: ok".into(),
            is_error: false,
        });
        t.push(Message::Assistant {
            content: "done".into(),
        });

        let json = serde_json::to_string(&t).unwrap();
        let back: Transcript = serde_json::from_str(&json).unwrap();
        assert_eq!(back, t);
        assert_eq!(back.messages().len(), 4);
        assert!(matches!(back.messages()[2], Message::ToolResult { .. }));
    }

    #[test]
    fn serialized_shape_is_readable() {
        let m = Message::ToolCall {
            call_id: "c9".into(),
            name: "edit".into(),
            arguments: serde_json::json!({}),
        };
        let v: serde_json::Value = serde_json::to_value(&m).unwrap();
        assert_eq!(v["role"], "toolCall"); // tag 字段存在
        assert_eq!(v["callId"], "c9");
    }
}
