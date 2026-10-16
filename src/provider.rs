//! Provider：对一个 LLM API 的适配层（GLOSSARY.md）。
//!
//! 本模块实现：
//! 1. 02 ticket 的 OpenAI 兼容 SSE 字节流 → 统一事件的纯函数解析器（ADR-0001：手写增量解析）；
//! 2. `Provider` trait：Transcript + tools 声明 → 统一事件流（本 ticket 的窄 seam 接口侧）；
//! 3. `OpenAiProvider`：reqwest 消费 chat/completions SSE，逐 chunk 喂解析器。
//!
//! trait 用 RPITIT（Rust 1.75+ 原生 async fn in trait）而非 #[async_trait]：
//! 目标 tokio 单运行时，无需 dyn 兼容（调用方 monomorphization 即可），
//! RPITIT 零装箱零动态分发，也少一个过程宏依赖。

use std::future::Future;
use std::pin::Pin;

use futures_util::StreamExt;
use serde_json::Value;

/// 流式接口产出的统一事件流（BoxStream，调用方无需关心底层 HTTP 细节）。
pub type EventStream = Pin<Box<dyn futures_util::Stream<Item = Result<ProviderEvent>> + Send + 'static>>;

/// 解析器产出的统一事件（spec：Provider 窄 seam 的输出侧）。
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// 文本增量
    TextDelta {
        delta: String,
    },
    /// ToolCall 增量。index 标识并行调用的槽位；id/name 通常只在首个碎片出现，
    /// 后续碎片为 None；参数为 JSON 字符串碎片，按序拼接得完整参数。
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        name: Option<String>,
        arguments_delta: Option<String>,
    },
    /// 完成信号（`[DONE]` 行触发），携带流中最后见到的 finish_reason。
    Done {
        finish_reason: Option<String>,
    },
}

/// SSE 字节流解析状态机。内部缓冲处理行跨 chunk 切断；
/// 逐字节喂入与整段喂入产出完全一致的事件序列（硬性验收）。
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
    last_finish_reason: Option<String>,
    done: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    /// 是否已消费到 [DONE]（终止信号）。供流式消费方判断正常收尾。
    pub fn is_done(&self) -> bool {
        self.done
    }

    /// 外部强制标记终止（流错误后不再继续拉取）。供 provider 流包装层在
    /// 发出终止性错误后调用，避免 unfold 状态机在 EOF/错误后无限循环。
    pub fn mark_done(&mut self) {
        self.done = true;
    }

    /// 喂入任意大小的字节片段，返回本次新产生的完整事件。
    pub fn feed(&mut self, chunk: &[u8]) -> Vec<Event> {
        let mut events = Vec::new();
        if self.done {
            return events;
        }
        self.buf.extend_from_slice(chunk);
        // 只处理以 \n 结尾的完整行，不完整的尾部留在缓冲等下一个 chunk
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line_bytes: Vec<u8> = self.buf.drain(..=pos).collect();
            // 行内不含裸 \n（JSON 字符串会转义），单行 lossy 转换安全
            let mut line = String::from_utf8_lossy(&line_bytes[..line_bytes.len() - 1]).into_owned();
            if line.ends_with('\r') {
                line.pop();
            }
            self.feed_line(&line, &mut events);
        }
        events
    }

    fn feed_line(&mut self, line: &str, events: &mut Vec<Event>) {
        // OpenAI 协议只关心 data 行；空行（事件分隔）、event:/id:/`:` 注释行忽略
        let Some(rest) = line.strip_prefix("data:") else {
            return;
        };
        // SSE 规范：冒号后至多一个可选空格（同时兼容无空格变体）
        let payload = rest.strip_prefix(' ').unwrap_or(rest);
        if payload.trim() == "[DONE]" {
            self.done = true;
            events.push(Event::Done {
                finish_reason: self.last_finish_reason.take(),
            });
            return;
        }
        // 非法 JSON 行跳过不 panic
        let Ok(value) = serde_json::from_str::<Value>(payload) else {
            return;
        };
        self.feed_chunk_json(&value, events);
    }

    fn feed_chunk_json(&mut self, value: &Value, events: &mut Vec<Event>) {
        // 结构不符（无 choices / 空数组）静默跳过
        let Some(choice) = value.get("choices").and_then(|c| c.get(0)) else {
            return;
        };
        if let Some(reason) = choice.get("finish_reason").and_then(|f| f.as_str()) {
            self.last_finish_reason = Some(reason.to_string());
        }
        let Some(delta) = choice.get("delta") else {
            return;
        };
        if let Some(content) = delta.get("content").and_then(|c| c.as_str()) {
            if !content.is_empty() {
                events.push(Event::TextDelta {
                    delta: content.to_string(),
                });
            }
        }
        let Some(tool_calls) = delta.get("tool_calls").and_then(|t| t.as_array()) else {
            return;
        };
        for tc in tool_calls {
            let index = tc.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let id = tc.get("id").and_then(|i| i.as_str()).map(String::from);
            let function = tc.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str())
                .map(String::from);
            let arguments_delta = function
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                .map(String::from);
            events.push(Event::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Provider trait 与 OpenAI 兼容实现（本 ticket）
// ---------------------------------------------------------------------------

/// 统一事件流的单项：解析事件 + 流级错误。
/// 错误做成枚举单项以便后续扩展（如 usage 事件），当前事件即 `Event`。
#[derive(Debug, Clone, PartialEq)]
pub enum ProviderEvent {
    Event(Event),
}

/// Provider 层错误。全部可匹配变体，不 panic（thiserror）。
#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    /// 非成功 HTTP 状态（含网关返回的错误体前 500 字节便于排错）
    #[error("provider returned HTTP {status}: {body}")]
    HttpStatus { status: u16, body: String },
    /// 请求构建或传输失败（连接拒绝、超时、TLS 等）
    #[error("request failed: {0}")]
    Request(#[from] reqwest::Error),
    /// 流在中断处结束：未收到 [DONE] 即 EOF
    #[error("stream ended before [DONE] (received {bytes} bytes)")]
    StreamTruncated { bytes: u64 },
}

pub type Result<T> = std::result::Result<T, ProviderError>;

use crate::config::Connection;
use crate::tool::Tool;
use crate::transcript::{Message, Transcript};

/// 对一个 LLM API 的适配层接口（GLOSSARY.md：Provider）。
///
/// 输入统一的 Transcript 与 tools 声明，输出统一事件流。
/// trait 方法用 RPITIT：返回类型由实现内联给出，调用方无需 dyn/装箱。
/// 选 RPITIT 而非 #[async_trait] 的理由：本 crate 目标 tokio 单运行时、
/// 静态分发足够（后续 agent loop 只接一个 provider 实例），省去 Box 装箱
/// 与过程宏依赖；若未来需要 dyn Provider（多 provider 运行时选择）再引入
/// async-trait 或手动 BoxStream 也不迟。
pub trait Provider {
    /// 发起一次流式补全请求，返回统一事件流。
    ///
    /// HTTP 状态错误在首个 poll 即给出（本方法内同步检查），流中断错误
    /// 以 `Err(ProviderError::StreamTruncated)` 作为流的最后一个 item。
    fn stream(&self, transcript: &Transcript, tools: &[Tool]) -> impl Future<Output = Result<EventStream>> + Send;
}

/// 统一 Message → OpenAI chat/completions 请求条目。
/// ToolCall 变体输出 role:assistant + tool_calls 数组；ToolResult 输出 role:tool + tool_call_id。
fn message_to_openai(message: &Message) -> Value {
    match message {
        Message::User { content } => serde_json::json!({ "role": "user", "content": content }),
        Message::Assistant { content } => serde_json::json!({ "role": "assistant", "content": content }),
        Message::ToolCall { call_id, name, arguments } => serde_json::json!({
            "role": "assistant",
            "content": null,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments.to_string(),
                }
            }]
        }),
        Message::ToolResult { call_id, content, is_error } => serde_json::json!({
            "role": "tool",
            "tool_call_id": call_id,
            // OpenAI 协议无 is_error 位；失败信息以文本形式让模型自行判断
            "content": if *is_error { format!("Error: {content}") } else { content.clone() },
        }),
    }
}

/// Tool 声明 → OpenAI function 格式（JSON Schema 参数）。
fn tool_to_openai(tool: &Tool) -> Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    })
}

/// 组装 chat/completions 请求体。
fn build_request_body(transcript: &Transcript, tools: &[Tool], model: &str) -> Value {
    serde_json::json!({
        "model": model,
        "messages": transcript.messages().iter().map(message_to_openai).collect::<Vec<_>>(),
        "tools": tools.iter().map(tool_to_openai).collect::<Vec<_>>(),
        "stream": true,
    })
}

/// OpenAI 兼容协议实现（GLOSSARY.md 第一版唯一 provider）。
/// endpoint 指向完整 chat/completions URL；Bearer 鉴权；SSE 流式响应喂给 SseParser。
pub struct OpenAiProvider {
    http: reqwest::Client,
    endpoint: String,
    api_key: String,
    model: String,
}

impl OpenAiProvider {
    pub fn new(connection: Connection) -> Self {
        let api_key = connection.api_key().to_string();
        Self {
            http: reqwest::Client::new(),
            endpoint: connection.endpoint,
            api_key,
            model: connection.model_id,
        }
    }
}

impl Provider for OpenAiProvider {
    fn stream(&self, transcript: &Transcript, tools: &[Tool]) -> impl Future<Output = Result<EventStream>> + Send {
        let body = build_request_body(transcript, tools, &self.model);
        let http = self.http.clone();
        let endpoint = self.endpoint.clone();
        let api_key = self.api_key.clone();

        async move {
            let response = http
                .post(&endpoint)
                .bearer_auth(&api_key)
                .json(&body)
                .send()
                .await?;
            let status = response.status();
            if !status.is_success() {
                let body = response.text().await.unwrap_or_default();
                let mut body = body;
                body.truncate(500);
                return Err(ProviderError::HttpStatus {
                    status: status.as_u16(),
                    body,
                });
            }

            let byte_stream = response.bytes_stream();
            let stream = futures_util::stream::unfold(
                (byte_stream, SseParser::new(), 0u64, std::collections::VecDeque::new()),
                |(mut byte_stream, mut parser, mut bytes_seen, mut pending)| async move {
                    loop {
                        // 先排空已解析事件，再拉新 chunk（同 chunk 多事件不丢失）
                        if let Some(event) = pending.pop_front() {
                            return Some((
                                Ok(ProviderEvent::Event(event)),
                                (byte_stream, parser, bytes_seen, pending),
                            ));
                        }
                        if parser.is_done() {
                            return None;
                        }
                        match byte_stream.next().await {
                            Some(Ok(chunk)) => {
                                bytes_seen += chunk.len() as u64;
                                let mut events = parser.feed(&chunk);
                                if !events.is_empty() {
                                    let first = events.remove(0);
                                    pending.extend(events);
                                    return Some((
                                        Ok(ProviderEvent::Event(first)),
                                        (byte_stream, parser, bytes_seen, pending),
                                    ));
                                }
                                // 本次 chunk 没产出完整事件，继续拉下一个 chunk
                            }
                            Some(Err(e)) => {
                                parser.mark_done();
                                return Some((
                                    Err(ProviderError::Request(e)),
                                    (byte_stream, parser, bytes_seen, pending),
                                ));
                            }
                            None => {
                                // EOF：正常收尾要求见过 [DONE]；否则流中断。
                                // 错误只发一次——发过后 mark_done，下一轮终止，
                                // 否则 byte_stream 反复 None 造成无限 Err 循环。
                                if parser.is_done() {
                                    return None;
                                }
                                parser.mark_done();
                                return Some((
                                    Err(ProviderError::StreamTruncated { bytes: bytes_seen }),
                                    (byte_stream, parser, bytes_seen, pending),
                                ));
                            }
                        }
                    }
                },
            );

            Ok(Box::pin(stream) as EventStream)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 收集一个解析器对若干 chunk 喂入产生的全部事件
    fn parse_all(parser: &mut SseParser, chunks: &[&[u8]]) -> Vec<Event> {
        let mut events = Vec::new();
        for chunk in chunks {
            events.extend(parser.feed(chunk));
        }
        events
    }

    fn feed_whole(fixture: &str) -> Vec<Event> {
        parse_all(&mut SseParser::new(), &[fixture.as_bytes()])
    }

    fn text_delta(s: &str) -> Event {
        Event::TextDelta {
            delta: s.to_string(),
        }
    }

    fn tool_delta(index: usize, id: Option<&str>, name: Option<&str>, args: Option<&str>) -> Event {
        Event::ToolCallDelta {
            index,
            id: id.map(String::from),
            name: name.map(String::from),
            arguments_delta: args.map(String::from),
        }
    }

    // ---------- fixture：正常文本流 ----------

    static FIXTURE_TEXT: &str = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"He"},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn text_stream_produces_text_deltas_then_done() {
        let events = feed_whole(FIXTURE_TEXT);
        assert_eq!(
            events,
            vec![text_delta("He"), text_delta("llo"), Event::Done { finish_reason: Some("stop".into()) }]
        );
    }

    // ---------- fixture：单个 tool call 流 ----------

    static FIXTURE_TOOL_SINGLE: &str = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read","arguments":""}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"pa"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"src/main.rs\"}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn single_tool_call_stream_carries_fragments() {
        let events = feed_whole(FIXTURE_TOOL_SINGLE);
        assert_eq!(
            events,
            vec![
                tool_delta(0, Some("call_1"), Some("read"), Some("")),
                tool_delta(0, None, None, Some("{\"pa")),
                tool_delta(0, None, None, Some("th\":\"src/main.rs\"}")),
                Event::Done {
                    finish_reason: Some("tool_calls".into())
                },
            ]
        );
        // 参数碎片按序拼回完整 JSON
        let joined: String = events
            .iter()
            .filter_map(|e| match e {
                Event::ToolCallDelta { arguments_delta, .. } => arguments_delta.clone(),
                _ => None,
            })
            .collect();
        let parsed: Value = serde_json::from_str(&joined).unwrap();
        assert_eq!(parsed["path"], "src/main.rs");
    }

    // ---------- fixture：多 tool call 并行流（index 0/1 交错） ----------

    static FIXTURE_TOOL_PARALLEL: &str = concat!(
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_a","function":{"name":"read","arguments":""}},{"index":1,"id":"call_b","function":{"name":"bash","arguments":""}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"{\"comm"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\":\"a.rs\"}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"function":{"arguments":"and\":\"ls\"}"}}]},"finish_reason":null}]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn parallel_tool_calls_keep_index_and_interleaving() {
        let events = feed_whole(FIXTURE_TOOL_PARALLEL);
        assert_eq!(
            events,
            vec![
                tool_delta(0, Some("call_a"), Some("read"), Some("")),
                tool_delta(1, Some("call_b"), Some("bash"), Some("")),
                tool_delta(1, None, None, Some("{\"comm")),
                tool_delta(0, None, None, Some("{\"path\":\"a.rs\"}")),
                tool_delta(1, None, None, Some("and\":\"ls\"}")),
                Event::Done {
                    finish_reason: Some("tool_calls".into())
                },
            ]
        );
    }

    // ---------- fixture：坏 JSON / 结构不符行 ----------

    static FIXTURE_BAD_LINES: &str = concat!(
        "data: {not json}\n\n",
        r#"data: {"error":{"message":"rate limited"}}"#,
        "\n\n",
        r#"data: {"choices":[]}"#,
        "\n\n",
        r#"data: {"choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":null}]}"#,
        "\n\n",
        "data: [DONE]\n\n",
    );

    #[test]
    fn bad_json_and_malformed_chunks_are_skipped_without_panic() {
        let events = feed_whole(FIXTURE_BAD_LINES);
        assert_eq!(
            events,
            vec![text_delta("ok"), Event::Done { finish_reason: None }]
        );
    }

    // ---------- 协议变体：无空格 data:、CRLF、注释行、多行同 chunk ----------

    #[test]
    fn nospace_data_prefix_crlf_and_comment_lines() {
        let fixture = concat!(
            ": keep-alive\n",
            "event: ping\n",
            r#"data:{"choices":[{"index":0,"delta":{"content":"a"},"finish_reason":null}]}"#,
            "\r\n",
            r#"data:{"choices":[{"index":0,"delta":{"content":"b"},"finish_reason":null}]}"#,
            "\r\n\r\n",
            "data:  [DONE]\r\n",
        );
        let events = feed_whole(fixture);
        assert_eq!(
            events,
            vec![
                text_delta("a"),
                text_delta("b"),
                Event::Done { finish_reason: None }
            ]
        );
    }

    // ---------- 硬性验收：逐字节喂入与整段喂入结果一致 ----------

    #[test]
    fn byte_by_byte_feeding_matches_whole_feeding() {
        for fixture in [
            FIXTURE_TEXT,
            FIXTURE_TOOL_SINGLE,
            FIXTURE_TOOL_PARALLEL,
            FIXTURE_BAD_LINES,
        ] {
            let whole = feed_whole(fixture);
            let bytes = fixture.as_bytes();
            let mut parser = SseParser::new();
            let mut bytewise = Vec::new();
            for b in bytes {
                bytewise.extend(parser.feed(&[*b]));
            }
            assert_eq!(bytewise, whole, "fixture diverges when fed byte-by-byte");
        }
    }

    #[test]
    fn arbitrary_chunk_cutting_matches_whole_feeding() {
        let fixture = FIXTURE_TOOL_PARALLEL;
        let whole = feed_whole(fixture);
        // 固定切法：按 7 字节切块（刻意不对齐行边界）
        let mut parser = SseParser::new();
        let mut cut = Vec::new();
        for chunk in fixture.as_bytes().chunks(7) {
            cut.extend(parser.feed(chunk));
        }
        assert_eq!(cut, whole);
    }

    // ---------- 杂项 ----------

    #[test]
    fn incomplete_trailing_line_is_buffered_until_newline_arrives() {
        let mut parser = SseParser::new();
        let half = r#"data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}"#;
        assert!(parser.feed(half.as_bytes()).is_empty()); // 无 \n，不产出
        let events = parser.feed(b"\n\ndata: [DONE]\n");
        assert_eq!(
            events,
            vec![text_delta("hi"), Event::Done { finish_reason: None }]
        );
    }

    #[test]
    fn feed_after_done_is_noop() {
        let mut parser = SseParser::new();
        let events = parser.feed(b"data: [DONE]\n");
        assert_eq!(events, vec![Event::Done { finish_reason: None }]);
        let after = parser.feed(b"data: {garbage}\n");
        assert!(after.is_empty());
    }

    #[test]
    fn empty_content_delta_produces_no_event() {
        let mut parser = SseParser::new();
        let events = parser.feed(concat!(
            r#"data: {"choices":[{"index":0,"delta":{"content":""},"finish_reason":null}]}"#,
            "\ndata: [DONE]\n"
        ).as_bytes());
        assert_eq!(events, vec![Event::Done { finish_reason: None }]);
    }

    // ---------- ticket 03 验收：mock server 集成测试 ----------

    use crate::config::Connection;
    use futures_util::StreamExt as _;

    /// 构造指向 mock server 的 Connection
    fn connection_to(uri: &str) -> Connection {
        Connection::with_parts("glm-5.3-flash", &format!("{uri}/v1/chat/completions"), "test-key")
    }

    fn transcript_with_user(text: &str) -> Transcript {
        let mut t = Transcript::new();
        t.push(Message::User { content: text.to_string() });
        t
    }

    async fn collect_stream(
        provider: &OpenAiProvider,
        transcript: &Transcript,
        tools: &[Tool],
    ) -> Vec<Result<ProviderEvent>> {
        let stream = provider.stream(transcript, tools).await.unwrap();
        futures_util::StreamExt::collect::<Vec<_>>(stream).await
    }

    #[tokio::test]
    async fn request_body_carries_model_messages_tools() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .and(wiremock::matchers::path("/v1/chat/completions"))
            .respond_with(|_req: &wiremock::Request| {
                // 记下请求体供断言（通过响应头带回是 hack；改为全局状态更清晰，
                // 但 wiremock 闭包要求 Fn，简单起见断言放响应体外——见下方 body 断言测试）
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(concat!(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":null}]}\n\n",
                        "data: [DONE]\n\n"
                    ))
                    .insert_header("x-echo-model", "seen")
            })
            .mount(&server)
            .await;

        let provider = OpenAiProvider::new(connection_to(&server.uri()));
        let tools = vec![Tool {
            name: "read".into(),
            description: "read a file".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"]
            }),
        }];
        let transcript = transcript_with_user("hi");
        let events = collect_stream(&provider, &transcript, &tools).await;

        // 流正常收尾
        assert!(server.received_requests().await.is_some());
        assert!(matches!(events.last(), Some(Ok(ProviderEvent::Event(Event::Done { .. })))));

        // 断言请求体：model / messages / tools / stream
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1);
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["model"], "glm-5.3-flash");
        assert_eq!(body["stream"], true);
        assert_eq!(body["messages"][0]["role"], "user");
        assert_eq!(body["messages"][0]["content"], "hi");
        assert_eq!(body["tools"][0]["type"], "function");
        assert_eq!(body["tools"][0]["function"]["name"], "read");
        assert_eq!(
            body["tools"][0]["function"]["parameters"]["required"][0],
            "file_path"
        );

        // Bearer 鉴权头
        let auth = requests[0]
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert!(auth.starts_with("Bearer "), "auth header: {auth}");
    }

    #[tokio::test]
    async fn recorded_sse_response_replays_as_events() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(|_req: &wiremock::Request| {
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(concat!(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"He\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"llo\"},\"finish_reason\":null}]}\n\n",
                        "data: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
                        "data: [DONE]\n\n"
                    ))
            })
            .mount(&server)
            .await;

        let provider = OpenAiProvider::new(connection_to(&server.uri()));
        let transcript = transcript_with_user("hi");
        let events = collect_stream(&provider, &transcript, &[]).await;

        let expected: Vec<Result<ProviderEvent>> = vec![
            Ok(ProviderEvent::Event(text_delta("He"))),
            Ok(ProviderEvent::Event(text_delta("llo"))),
            Ok(ProviderEvent::Event(Event::Done { finish_reason: Some("stop".into()) })),
        ];
        assert_eq!(events.len(), expected.len());
        for (got, want) in events.iter().zip(expected.iter()) {
            assert_eq!(got.as_ref().unwrap(), want.as_ref().unwrap());
        }
    }

    #[tokio::test]
    async fn http_error_status_becomes_http_status_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(wiremock::ResponseTemplate::new(401).set_body_string("bad key"))
            .mount(&server)
            .await;

        let provider = OpenAiProvider::new(connection_to(&server.uri()));
        let transcript = transcript_with_user("hi");
        let result = provider.stream(&transcript, &[]).await;

        match result {
            Err(ProviderError::HttpStatus { status, body }) => {
                assert_eq!(status, 401);
                assert!(body.contains("bad key"));
            }
            Err(_) => panic!("expected HttpStatus error"),
            Ok(_) => panic!("expected HttpStatus error, got Ok"),
        }
    }

    #[tokio::test]
    async fn truncated_stream_yields_stream_truncated_error() {
        let server = wiremock::MockServer::start().await;
        wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(|_req: &wiremock::Request| {
                // 只有文本增量，没有 [DONE] 就断流
                wiremock::ResponseTemplate::new(200)
                    .set_body_string(concat!(
                        "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"partial\"},\"finish_reason\":null}]}\n\n"
                    ))
            })
            .mount(&server)
            .await;

        let provider = OpenAiProvider::new(connection_to(&server.uri()));
        let transcript = transcript_with_user("hi");
        let events = collect_stream(&provider, &transcript, &[]).await;

        assert!(matches!(
            events.last(),
            Some(Err(ProviderError::StreamTruncated { .. }))
        ));
    }

    #[tokio::test]
    async fn tool_call_messages_round_trip_through_request_body() {
        let server = wiremock::MockServer::start().await;
        let mocked = wiremock::Mock::given(wiremock::matchers::method("POST"))
            .respond_with(|_req: &wiremock::Request| {
                wiremock::ResponseTemplate::new(200)
                    .set_body_string("data: [DONE]\n\n")
            })
            .mount(&server)
            .await;

        let provider = OpenAiProvider::new(connection_to(&server.uri()));
        let mut transcript = Transcript::new();
        transcript.push(Message::User { content: "list files".into() });
        transcript.push(Message::ToolCall {
            call_id: "call_9".into(),
            name: "read".into(),
            arguments: serde_json::json!({"file_path": "a.rs"}),
        });
        transcript.push(Message::ToolResult {
            call_id: "call_9".into(),
            content: "contents…".into(),
            is_error: false,
        });

        let _ = collect_stream(&provider, &transcript, &[]).await;

        let requests = server.received_requests().await.unwrap();
        let body: Value = serde_json::from_slice(&requests[0].body).unwrap();
        assert_eq!(body["messages"][1]["role"], "assistant");
        assert_eq!(body["messages"][1]["tool_calls"][0]["id"], "call_9");
        assert_eq!(body["messages"][1]["tool_calls"][0]["function"]["name"], "read");
        assert_eq!(body["messages"][2]["role"], "tool");
        assert_eq!(body["messages"][2]["tool_call_id"], "call_9");
    }
}
