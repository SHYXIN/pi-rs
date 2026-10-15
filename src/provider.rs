//! Provider：对一个 LLM API 的适配层（GLOSSARY.md）。
//!
//! 本模块当前实现 OpenAI 兼容协议的 SSE 字节流 → 统一事件的纯函数解析器
//! （ADR-0001：手写增量解析，不引 eventsource 库）。HTTP 请求在后续 ticket。

use serde_json::Value;

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
}
