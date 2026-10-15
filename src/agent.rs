//! Agent loop：驱动一次会话的核心循环（GLOSSARY.md）。
//!
//! 流程：Transcript → Provider 流式回复 → 无工具调用则终止并返回最终回复；
//! 有工具调用 → 逐个校验/执行（tokio 并发 join，结果按 ToolCall 顺序回填
//! Transcript）→ 再次调用 Provider，循环往复直到给出无工具调用的最终回复。
//!
//! beforeToolCall 拦截点：注册闭包，对每次工具调用返回放行与否（默认放行）。
//! 语义对齐 pi 的同名 hook；为后续权限系统预留 seam（届时把"放行/拒绝"的
//! 决策来源从闭包换成权限服务即可，循环侧接口不变）。
//!
//! 测试策略：脚本化 FakeProvider 直接实现 `Provider` trait，重放预编排的
//! 事件序列，不碰网络。放在本文件 `#[cfg(test)]` 而非独立 test-util 文件：
//! 它是循环测试的专属脚手架（依赖事件构造 helper），与 loop 同文件内聚最简，
//! 未来若多模块复用再拆出。

use std::sync::Arc;

use futures_util::StreamExt;

use crate::provider::{Event, Provider, ProviderError, ProviderEvent};
use crate::tool::{ToolContext, ToolOutput, ToolRegistry};
use crate::transcript::{Message, Transcript};

/// 单次会话允许的最大循环轮数（一次 provider 调用 + 工具执行 = 一轮）。
///
/// 取值理由：正常任务几轮内收敛；设 32 的目的是只防「模型反复调用工具
/// 永不收尾」的失控场景，而非限制正常多步任务。到顶时终止循环并返回
/// 可匹配的错误，不 panic、不静默截断。
pub const MAX_ITERATIONS: u32 = 32;

/// Agent loop 层错误。
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// provider 流或请求失败，原样透传
    #[error(transparent)]
    Provider(#[from] ProviderError),
    /// 达到最大循环轮数仍未给出最终回复（防失控的安全阀）
    #[error("agent loop exceeded {max} iterations without a final reply")]
    MaxIterations { max: u32 },
    /// beforeToolCall 拦截点拒绝了本次工具调用
    #[error("tool call `{name}` denied by beforeToolCall hook: {reason}")]
    DeniedByHook { name: String, reason: String },
}

pub type Result<T> = std::result::Result<T, AgentError>;

/// beforeToolCall 拦截点的判定结果。
#[derive(Debug, Clone, PartialEq)]
pub enum HookDecision {
    /// 放行（默认行为）
    Allow,
    /// 拒绝执行；reason 会作为错误信息返回（后续权限系统可改为回传模型）
    Deny { reason: String },
}

/// beforeToolCall 闭包签名：收到一次即将执行的工具调用，返回放行与否。
///
/// 用 `Arc<dyn Fn>` 而非泛型参数：循环结构体字段要存它，泛型会传染到
/// 所有使用者；这里调用频率低、无热路径性能诉求，动态分发可接受。
pub type BeforeToolCallHook = Arc<dyn Fn(&str, &serde_json::Value) -> HookDecision + Send + Sync>;

/// 一次 loop 运行的产出：最终回复文本 + 完整 Transcript（含全部中间消息）。
#[derive(Debug, Clone, PartialEq)]
pub struct AgentRun {
    /// 最后一条 assistant 文本回复（终止条件：该回复不带工具调用）
    pub final_reply: String,
    /// 本轮会话的完整消息历史（与调用方传入并被追加后的 transcript 一致）
    pub transcript: Transcript,
}

/// 单轮回复的中间形态：拼接后的文本 + 聚合完成的工具调用（可能为空）。
struct PendingReply {
    text: String,
    /// None：本轮回复没有工具调用（终止信号）；Some：聚合好的调用列表
    tool_calls: Option<Vec<PendingToolCall>>,
}

struct PendingToolCall {
    call_id: String,
    name: String,
    arguments: serde_json::Value,
}

/// 中央 Agent Loop（GLOSSARY.md）。
///
/// 泛型参数 P 为 provider 实现（RPITIT 静态分发，见 provider.rs 注释）。
pub struct AgentLoop<'a, P: Provider> {
    provider: &'a P,
    registry: &'a ToolRegistry,
    ctx: ToolContext,
    before_tool_call: Option<BeforeToolCallHook>,
}

impl<'a, P: Provider> AgentLoop<'a, P> {
    pub fn new(provider: &'a P, registry: &'a ToolRegistry, ctx: ToolContext) -> Self {
        Self {
            provider,
            registry,
            ctx,
            before_tool_call: None,
        }
    }

    /// 注册 beforeToolCall 拦截闭包（语义对齐 pi 同名 hook）。
    /// 未注册时默认放行所有工具调用。
    pub fn with_before_tool_call(mut self, hook: BeforeToolCallHook) -> Self {
        self.before_tool_call = Some(hook);
        self
    }

    /// 驱动一次会话直到模型给出无工具调用的最终回复。
    ///
    /// transcript 会被原地追加（assistant 回复、tool call、tool result）。
    pub async fn run(&self, transcript: &mut Transcript) -> Result<AgentRun> {
        let tools = self.registry.tools();

        for _ in 0..MAX_ITERATIONS {
            // —— 1. 调 provider，收集完整回复（文本增量拼接 + 工具调用聚合）——
            let reply = self.collect_reply(transcript, &tools).await?;

            // —— 2. 无工具调用 → 终止，返回最终回复 ——
            let Some(tool_calls) = reply.tool_calls else {
                transcript.push(Message::Assistant {
                    content: reply.text.clone(),
                });
                return Ok(AgentRun {
                    final_reply: reply.text,
                    transcript: transcript.clone(),
                });
            };

            // —— 3. 有工具调用：先落 assistant 文本（若非空），再落 tool_call 消息 ——
            // OpenAI 协议中 assistant 消息可同时携带 content 与 tool_calls；
            // Message 模型把两者拆成独立变体，文本先于 ToolCall 入账保持时序，
            // 下一轮 provider 序列化时模型能看到自己说过的话。
            if !reply.text.is_empty() {
                transcript.push(Message::Assistant { content: reply.text });
            }
            for call in &tool_calls {
                transcript.push(Message::ToolCall {
                    call_id: call.call_id.clone(),
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });
            }

            // —— 4. beforeToolCall 拦截（顺序检查；拒绝即终止本轮会话）——
            if let Some(hook) = &self.before_tool_call {
                for call in &tool_calls {
                    match hook(&call.name, &call.arguments) {
                        HookDecision::Allow => {}
                        HookDecision::Deny { reason } => {
                            return Err(AgentError::DeniedByHook {
                                name: call.name.clone(),
                                reason,
                            });
                        }
                    }
                }
            }

            // —— 5. 并发执行所有工具调用，结果按原顺序回填 ——
            // 每个调用独立校验 + 执行；校验失败/执行失败/未知工具都转成
            // is_error=true 的 ToolResult 回传模型（模型可自行纠正重试），
            // 循环不因此中断——只有 provider 层错误才跳出循环。
            let executions = tool_calls.iter().map(|call| {
                let output = self.registry.execute(&call.name, call.arguments.clone(), &self.ctx);
                async move {
                    let result = match output.await {
                        Ok(ToolOutput::Success(text)) => Message::ToolResult {
                            call_id: call.call_id.clone(),
                            content: text,
                            is_error: false,
                        },
                        Ok(ToolOutput::ValidationFailed(msg)) => Message::ToolResult {
                            call_id: call.call_id.clone(),
                            content: msg,
                            is_error: true,
                        },
                        Ok(ToolOutput::ExecutionError(msg)) => Message::ToolResult {
                            call_id: call.call_id.clone(),
                            content: msg,
                            is_error: true,
                        },
                        // 未知工具名等注册表级错误：同样包装为错误 ToolResult
                        Err(e) => Message::ToolResult {
                            call_id: call.call_id.clone(),
                            content: e.to_string(),
                            is_error: true,
                        },
                    };
                    (call.call_id.clone(), result)
                }
            });
            let results = futures_util::future::join_all(executions).await;

            // 按工具调用顺序把结果追加进 transcript（校验失败也占一条，
            // 保证 tool_call 与 tool_result 一一对应，满足 OpenAI 协议约束）
            for (_, message) in results {
                transcript.push(message);
            }
        }

        Err(AgentError::MaxIterations { max: MAX_ITERATIONS })
    }

    /// 消费一次 provider 流：拼接文本增量、按 index 聚合工具调用碎片。
    ///
    /// 流式协议中同一个工具调用的 id/name/参数分多个 delta 到达（见
    /// provider.rs 的 Event::ToolCallDelta 注释），这里按 index 归并：
    /// id/name 取首个非空出现，参数字符串按序拼接后整体 parse。
    async fn collect_reply(
        &self,
        transcript: &Transcript,
        tools: &[crate::tool::Tool],
    ) -> Result<PendingReply> {
        let stream = self.provider.stream(transcript, tools).await?;
        futures_util::pin_mut!(stream);

        let mut text = String::new();
        // index → (call_id, name, arguments_buffer)；BTreeMap 保 key 序稳定，测试断言可预测
        let mut slots: std::collections::BTreeMap<usize, (Option<String>, Option<String>, String)> =
            std::collections::BTreeMap::new();

        while let Some(item) = stream.next().await {
            match item? {
                ProviderEvent::Event(Event::TextDelta { delta }) => text.push_str(&delta),
                ProviderEvent::Event(Event::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                }) => {
                    let slot = slots.entry(index).or_default();
                    if slot.0.is_none() {
                        slot.0 = id;
                    }
                    if slot.1.is_none() {
                        slot.1 = name;
                    }
                    if let Some(frag) = arguments_delta {
                        slot.2.push_str(&frag);
                    }
                }
                ProviderEvent::Event(Event::Done { .. }) => break,
            }
        }

        let tool_calls = if slots.is_empty() {
            None
        } else {
            let mut calls = Vec::with_capacity(slots.len());
            for (_, (call_id, name, args_buf)) in slots {
                // 参数碎片理论上拼出合法 JSON；坏 JSON 不 panic——
                // 把原始碎片文本作为字符串参数兜底，让下游校验给出可回传模型的错误
                let arguments: serde_json::Value = serde_json::from_str(&args_buf)
                    .unwrap_or_else(|_| serde_json::Value::String(args_buf.clone()));
                calls.push(PendingToolCall {
                    call_id: call_id.unwrap_or_default(),
                    name: name.unwrap_or_default(),
                    arguments,
                });
            }
            Some(calls)
        };

        Ok(PendingReply { text, tool_calls })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::EventStream;
    use crate::tool::Tool;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    // ------------------------------------------------------------------
    // FakeProvider：脚本化 provider，按调用次数重放预编排的回复脚本。
    // 不碰网络（ticket 验收要求）。实现与 OpenAI 相同的 Provider trait，
    // 验证 loop 只依赖 trait seam，不感知具体 provider。
    // ------------------------------------------------------------------

    /// 一次脚本化的 provider 回复
    #[derive(Debug, Clone)]
    enum Script {
        /// 纯文本回复（无工具调用）→ 终止循环
        Text(&'static str),
        /// 文本 + 若干工具调用（模拟流式碎片：id/name/参数拆成多段 delta 重放）
        ToolCalls {
            text: &'static str,
            calls: Vec<ScriptedCall>,
        },
    }

    #[derive(Debug, Clone)]
    struct ScriptedCall {
        call_id: &'static str,
        name: &'static str,
        /// 完整参数 JSON 文本（合法与否均可——坏 JSON 用于测兜底路径）
        arguments_json: String,
    }

    fn call(call_id: &'static str, name: &'static str, args: serde_json::Value) -> ScriptedCall {
        ScriptedCall {
            call_id,
            name,
            arguments_json: args.to_string(),
        }
    }

    /// 并行性测试用的可读性标注：延迟由注册表里的慢速执行器（EchoExecutor.delay_ms）
    /// 承载，脚本侧只负责让 slow_echo 出现在调用列表首位。
    fn slow_call(call_id: &'static str, name: &'static str, args: serde_json::Value) -> ScriptedCall {
        call(call_id, name, args)
    }

    impl Provider for FakeProvider {
        type Stream =
            Pin<Box<dyn futures_util::Stream<Item = crate::provider::Result<ProviderEvent>> + Send>>;

        fn stream(
            &self,
            transcript: &Transcript,
            _tools: &[Tool],
        ) -> impl Future<Output = crate::provider::Result<EventStream>> + Send {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            self.seen_transcripts.lock().unwrap().push(transcript.clone());
            let step = self
                .script
                .get(n)
                .unwrap_or_else(|| panic!("FakeProvider script exhausted at call {n}"));
            let events: Vec<crate::provider::Result<ProviderEvent>> = match step {
                Script::Text(t) => vec![
                    Ok(ProviderEvent::Event(Event::TextDelta { delta: t.to_string() })),
                    Ok(ProviderEvent::Event(Event::Done {
                        finish_reason: Some("stop".into()),
                    })),
                ],
                Script::ToolCalls { text, calls } => {
                    let mut events = vec![Ok(ProviderEvent::Event(Event::TextDelta {
                        delta: text.to_string(),
                    }))];
                    for (i, c) in calls.iter().enumerate() {
                        // 模拟真实流式碎片：首个 delta 带 id+name，参数拆两段按序到达
                        let (a, b) = c.arguments_json.split_at(c.arguments_json.len() / 2);
                        events.push(Ok(ProviderEvent::Event(Event::ToolCallDelta {
                            index: i,
                            id: Some(c.call_id.to_string()),
                            name: Some(c.name.to_string()),
                            arguments_delta: Some(a.to_string()),
                        })));
                        events.push(Ok(ProviderEvent::Event(Event::ToolCallDelta {
                            index: i,
                            id: None,
                            name: None,
                            arguments_delta: Some(b.to_string()),
                        })));
                    }
                    events.push(Ok(ProviderEvent::Event(Event::Done {
                        finish_reason: Some("tool_calls".into()),
                    })));
                    events
                }
            };
            let stream = futures_util::stream::iter(events);
            async move { Ok(Box::pin(stream) as EventStream) }
        }
    }

    /// FakeProvider：按 stream() 调用次序吐出脚本，脚本耗尽即 panic（测试脚本写错）。
    /// 记录每次收到的 transcript 快照，供断言「模型看到了什么」。
    struct FakeProvider {
        script: Vec<Script>,
        call_count: AtomicUsize,
        seen_transcripts: Mutex<Vec<Transcript>>,
    }

    impl FakeProvider {
        fn new(script: Vec<Script>) -> Self {
            Self {
                script,
                call_count: AtomicUsize::new(0),
                seen_transcripts: Mutex::new(Vec::new()),
            }
        }

        fn call_count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    // ------------------------------------------------------------------
    // 测试用工具：echo 回声执行器，可选延迟（验证并行性）
    // ------------------------------------------------------------------

    /// 回声工具：Success(args["echo"])。delay_ms 模拟慢工具。
    struct EchoExecutor {
        name: String,
        delay_ms: u64,
    }

    #[async_trait::async_trait]
    impl crate::tool::ToolExecutor for EchoExecutor {
        fn name(&self) -> &str {
            &self.name
        }

        async fn execute(&self, args: serde_json::Value, _ctx: &ToolContext) -> ToolOutput {
            if self.delay_ms > 0 {
                tokio::time::sleep(std::time::Duration::from_millis(self.delay_ms)).await;
            }
            ToolOutput::Success(
                args.get("echo").and_then(|v| v.as_str()).unwrap_or("").to_string(),
            )
        }
    }

    /// 注册 echo（立即）与 slow_echo（延迟 80ms）两个工具。
    fn echo_registry() -> ToolRegistry {
        let mut reg = ToolRegistry::new();
        let schema = || {
            serde_json::json!({
                "type": "object",
                "properties": {"echo": {"type": "string"}},
                "required": ["echo"]
            })
        };
        reg.register(
            Tool { name: "echo".into(), description: "echo back".into(), parameters: schema() },
            Arc::new(EchoExecutor { name: "echo".into(), delay_ms: 0 }),
        )
        .unwrap();
        reg.register(
            Tool {
                name: "slow_echo".into(),
                description: "echo back slowly".into(),
                parameters: schema(),
            },
            Arc::new(EchoExecutor { name: "slow_echo".into(), delay_ms: 80 }),
        )
        .unwrap();
        reg
    }

    fn ctx() -> ToolContext {
        ToolContext::new(".")
    }

    fn transcript_with_user(text: &str) -> Transcript {
        let mut t = Transcript::new();
        t.push(Message::User { content: text.to_string() });
        t
    }

    // ------------------------------------------------------------------
    // 验收 1：终止条件——无工具调用的回复直接返回
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn reply_without_tool_call_terminates_and_returns_final_reply() {
        let provider = FakeProvider::new(vec![Script::Text("all done")]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut transcript = transcript_with_user("hi");
        let run = loop_.run(&mut transcript).await.unwrap();

        assert_eq!(run.final_reply, "all done");
        assert_eq!(provider.call_count(), 1);
        // transcript: user + assistant 终止回复
        assert_eq!(run.transcript.messages().len(), 2);
        assert_eq!(
            run.transcript.messages()[1],
            Message::Assistant { content: "all done".into() }
        );
    }

    // ------------------------------------------------------------------
    // 验收 2：单工具调用 → 执行 → 结果回传 → 下轮终止
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn single_tool_call_executes_then_loop_continues_to_final_reply() {
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "let me check",
                calls: vec![call("call_1", "echo", serde_json::json!({"echo": "ping"}))],
            },
            Script::Text("got ping"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut transcript = transcript_with_user("say ping");
        let run = loop_.run(&mut transcript).await.unwrap();

        assert_eq!(run.final_reply, "got ping");
        assert_eq!(provider.call_count(), 2, "loop must continue after tool call");

        let msgs = run.transcript.messages();
        assert_eq!(msgs.len(), 5); // user, assistant(1st), toolCall, toolResult, assistant(final)
        assert_eq!(msgs[1], Message::Assistant { content: "let me check".into() });
        assert_eq!(
            msgs[2],
            Message::ToolCall {
                call_id: "call_1".into(),
                name: "echo".into(),
                arguments: serde_json::json!({"echo": "ping"}),
            }
        );
        assert_eq!(
            msgs[3],
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "ping".into(),
                is_error: false,
            }
        );
        assert_eq!(msgs[4], Message::Assistant { content: "got ping".into() });

        // 第二轮模型看到了完整历史（含 toolResult）
        let seen = provider.seen_transcripts.lock().unwrap();
        assert_eq!(seen[1].messages().len(), 4);
        assert_eq!(
            seen[1].messages()[3],
            Message::ToolResult {
                call_id: "call_1".into(),
                content: "ping".into(),
                is_error: false,
            }
        );
    }

    // ------------------------------------------------------------------
    // 验收 3：多工具并发执行，结果按 ToolCall 顺序回填
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn multiple_tool_calls_run_in_parallel_and_results_append_in_order() {
        // slow_echo 先发出、echo 后发出：串行执行总时长 ≥80ms；
        // 并发执行 <80ms（echo 不等 slow_echo）。结果顺序仍按调用顺序。
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "two calls",
                calls: vec![
                    slow_call("call_slow", "slow_echo", serde_json::json!({"echo": "S"})),
                    call("call_fast", "echo", serde_json::json!({"echo": "F"})),
                ],
            },
            Script::Text("both done"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let started = std::time::Instant::now();
        let mut transcript = transcript_with_user("go");
        let run = loop_.run(&mut transcript).await.unwrap();
        let elapsed = started.elapsed();

        let msgs = run.transcript.messages();
        // user, a("two calls"), toolCall(slow), toolCall(fast), toolResult(slow), toolResult(fast)
        assert_eq!(msgs[1], Message::Assistant { content: "two calls".into() });
        assert_eq!(
            msgs[4],
            Message::ToolResult {
                call_id: "call_slow".into(),
                content: "S".into(),
                is_error: false,
            }
        );
        assert_eq!(
            msgs[5],
            Message::ToolResult {
                call_id: "call_fast".into(),
                content: "F".into(),
                is_error: false,
            }
        );

        // 并行断言：串行下限为 sum(80,0)=80ms；并发放宽到 150ms 容忍调度抖动。
        assert!(
            elapsed < std::time::Duration::from_millis(150),
            "tool calls appear serialized: {elapsed:?}"
        );
    }

    // ------------------------------------------------------------------
    // 验收 4：校验失败 → 错误回传模型，循环继续（重试路径）
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn validation_failure_returns_error_tool_result_and_loop_continues() {
        // 第一轮：echo 参数缺 required 字段 → ValidationFailed
        // 第二轮：参数正确 → Success；第三轮：终止
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "oops missing field",
                calls: vec![call("call_bad", "echo", serde_json::json!({}))],
            },
            Script::ToolCalls {
                text: "retrying",
                calls: vec![call("call_good", "echo", serde_json::json!({"echo": "ok"}))],
            },
            Script::Text("recovered"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut transcript = transcript_with_user("echo please");
        let run = loop_.run(&mut transcript).await.unwrap();

        assert_eq!(run.final_reply, "recovered");
        assert_eq!(provider.call_count(), 3);

        let msgs = run.transcript.messages();
        // user, a("oops..."), toolCall(bad), toolResult(bad) —— 第一轮
        assert_eq!(
            msgs[3],
            Message::ToolResult {
                call_id: "call_bad".into(),
                content: "invalid arguments for tool `echo`: missing required field `echo` (type: string)"
                    .into(),
                is_error: true,
            }
        );
        // 第二轮：a("retrying"), toolCall(good), toolResult(good) —— 索引 4..6
        assert_eq!(
            msgs[6],
            Message::ToolResult {
                call_id: "call_good".into(),
                content: "ok".into(),
                is_error: false,
            }
        );
    }

    // ------------------------------------------------------------------
    // 验收 5：执行错误 / 未知工具 / 畸形参数 JSON → 错误 ToolResult，循环不断
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn execution_error_returns_error_tool_result_without_breaking_loop() {
        // read 一个不存在的文件 → ExecutionError（真实工具走完整链路）
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "read it",
                calls: vec![call(
                    "call_r",
                    "read",
                    serde_json::json!({"file_path": "no/such/file.txt"}),
                )],
            },
            Script::Text("saw the error"),
        ]);
        let mut registry = ToolRegistry::new();
        registry
            .register(
                Tool {
                    name: "read".into(),
                    description: "read".into(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {"file_path": {"type": "string"}},
                        "required": ["file_path"]
                    }),
                },
                Arc::new(crate::tool::ReadTool),
            )
            .unwrap();
        let loop_ = AgentLoop::new(&provider, &registry, ToolContext::new("."));

        let mut transcript = transcript_with_user("read");
        let run = loop_.run(&mut transcript).await.unwrap();

        assert_eq!(run.final_reply, "saw the error");
        let msgs = run.transcript.messages();
        match &msgs[3] {
            Message::ToolResult { call_id, content, is_error } => {
                assert_eq!(call_id, "call_r");
                assert!(is_error);
                assert!(content.contains("no/such/file.txt"), "{content}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unknown_tool_name_becomes_error_tool_result() {
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "calling something unregistered",
                calls: vec![call("call_u", "nonexistent", serde_json::json!({}))],
            },
            Script::Text("learned it is unknown"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut transcript = transcript_with_user("go");
        let run = loop_.run(&mut transcript).await.unwrap();

        assert_eq!(run.final_reply, "learned it is unknown");
        match &run.transcript.messages()[3] {
            Message::ToolResult { content, is_error, .. } => {
                assert!(is_error);
                assert!(content.contains("unknown tool"), "{content}");
                assert!(content.contains("nonexistent"), "{content}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn malformed_arguments_json_becomes_error_tool_result_not_panic() {
        // 参数碎片拼出坏 JSON：兜底成字符串参数 → 校验失败 → 错误 ToolResult
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "",
                calls: vec![ScriptedCall {
                    call_id: "c_bad",
                    name: "echo",
                    arguments_json: "{\"echo\": broken".to_string(),
                }],
            },
            Script::Text("survived bad json"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut t = transcript_with_user("go");
        let run = loop_.run(&mut t).await.unwrap();
        assert_eq!(run.final_reply, "survived bad json");
        match &run.transcript.messages()[2] {
            Message::ToolResult { content, is_error, .. } => {
                assert!(is_error);
                assert!(content.contains("must be a JSON object"), "{content}");
            }
            other => panic!("expected ToolResult, got {other:?}"),
        }
    }

    // ------------------------------------------------------------------
    // 验收 6：beforeToolCall hook——默认放行、可注册、可拒绝
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn hook_absent_by_default_allows_everything() {
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "",
                calls: vec![call("c1", "echo", serde_json::json!({"echo": "x"}))],
            },
            Script::Text("end"),
        ]);
        let registry = echo_registry();
        // 未注册 hook → 默认放行
        let loop_ = AgentLoop::new(&provider, &registry, ctx());
        let mut t = transcript_with_user("go");
        let run = loop_.run(&mut t).await.unwrap();
        assert_eq!(run.final_reply, "end");
    }

    #[tokio::test]
    async fn hook_can_inspect_call_and_allow() {
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "",
                calls: vec![call("c1", "echo", serde_json::json!({"echo": "secret"}))],
            },
            Script::Text("allowed once"),
        ]);
        let registry = echo_registry();

        let observed: Arc<Mutex<Vec<(String, serde_json::Value)>>> = Arc::new(Mutex::new(Vec::new()));
        let observed_in_hook = Arc::clone(&observed);
        let loop_ = AgentLoop::new(&provider, &registry, ctx()).with_before_tool_call(Arc::new(
            move |name: &str, args: &serde_json::Value| {
                observed_in_hook
                    .lock()
                    .unwrap()
                    .push((name.to_string(), args.clone()));
                HookDecision::Allow
            },
        ));
        let mut t = transcript_with_user("go");
        let run = loop_.run(&mut t).await.unwrap();

        assert_eq!(run.final_reply, "allowed once");
        assert_eq!(
            *observed.lock().unwrap(),
            vec![("echo".to_string(), serde_json::json!({"echo": "secret"}))]
        );
    }

    #[tokio::test]
    async fn hook_deny_stops_loop_with_denied_by_hook_error() {
        let provider = FakeProvider::new(vec![Script::ToolCalls {
            text: "",
            calls: vec![call("c1", "echo", serde_json::json!({"echo": "x"}))],
        }]);
        let registry = echo_registry();

        let loop_ = AgentLoop::new(&provider, &registry, ctx()).with_before_tool_call(Arc::new(
            |_name: &str, _args: &serde_json::Value| HookDecision::Deny {
                reason: "policy: echo is banned".into(),
            },
        ));
        let mut t = transcript_with_user("go");
        let err = loop_.run(&mut t).await.unwrap_err();

        match err {
            AgentError::DeniedByHook { name, reason } => {
                assert_eq!(name, "echo");
                assert_eq!(reason, "policy: echo is banned");
            }
            other => panic!("expected DeniedByHook, got {other:?}"),
        }
        // 工具未被执行（transcript 只到 toolCall，没有 toolResult）
        let msgs = t.messages();
        assert!(matches!(msgs.last(), Some(Message::ToolCall { .. })));
        assert!(!msgs.iter().any(|m| matches!(m, Message::ToolResult { .. })));
    }

    // ------------------------------------------------------------------
    // 验收 7：多轮直到终止（tool → tool → text）
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn multi_turn_tool_chain_until_termination() {
        let provider = FakeProvider::new(vec![
            Script::ToolCalls {
                text: "step 1",
                calls: vec![call("c1", "echo", serde_json::json!({"echo": "one"}))],
            },
            Script::ToolCalls {
                text: "step 2",
                calls: vec![call("c2", "echo", serde_json::json!({"echo": "two"}))],
            },
            Script::Text("chain finished"),
        ]);
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut t = transcript_with_user("run chain");
        let run = loop_.run(&mut t).await.unwrap();

        assert_eq!(run.final_reply, "chain finished");
        assert_eq!(provider.call_count(), 3);
        let msgs = run.transcript.messages();
        // user, a1, tc1, tr1, a2, tc2, tr2, final
        assert_eq!(msgs.len(), 8);
        assert_eq!(
            msgs[6],
            Message::ToolResult { call_id: "c2".into(), content: "two".into(), is_error: false }
        );
    }

    // ------------------------------------------------------------------
    // 安全阀与错误透传
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn exceeds_max_iterations_returns_max_iterations_error() {
        // 脚本一直给工具调用 → 触发 MAX_ITERATIONS 安全阀
        let infinite = || Script::ToolCalls {
            text: "",
            calls: vec![call("cx", "echo", serde_json::json!({"echo": "loop"}))],
        };
        let provider = FakeProvider::new((0..MAX_ITERATIONS + 2).map(|_| infinite()).collect());
        let registry = echo_registry();
        let loop_ = AgentLoop::new(&provider, &registry, ctx());

        let mut t = transcript_with_user("loop forever");
        let err = loop_.run(&mut t).await.unwrap_err();
        match err {
            AgentError::MaxIterations { max } => assert_eq!(max, MAX_ITERATIONS),
            other => panic!("expected MaxIterations, got {other:?}"),
        }
        // 每轮 2 条消息（toolCall + toolResult）+ 首条 user，恰好跑满上限
        assert_eq!(t.messages().len(), 1 + MAX_ITERATIONS as usize * 2);
    }

    #[tokio::test]
    async fn provider_error_propagates_out_of_loop() {
        struct BrokenProvider;
        impl Provider for BrokenProvider {
            type Stream = Pin<
                Box<dyn futures_util::Stream<Item = crate::provider::Result<ProviderEvent>> + Send>,
            >;
            fn stream(
                &self,
                _transcript: &Transcript,
                _tools: &[Tool],
            ) -> impl Future<Output = crate::provider::Result<EventStream>> + Send {
                let events: Vec<crate::provider::Result<ProviderEvent>> = vec![Err(
                    crate::provider::ProviderError::StreamTruncated { bytes: 7 },
                )];
                async move { Ok(Box::pin(futures_util::stream::iter(events)) as EventStream) }
            }
        }

        let registry = echo_registry();
        let loop_ = AgentLoop::new(&BrokenProvider, &registry, ctx());
        let mut t = transcript_with_user("go");
        let err = loop_.run(&mut t).await.unwrap_err();
        assert!(matches!(err, AgentError::Provider(_)), "{err:?}");
    }
}
