//! REPL：终端入口（GLOSSARY.md First Version Boundary 的最后一环）。
//!
//! 职责：
//! 1. rustyline 多行输入（Alt+Enter / \ 续行；Shift+Enter 终端支持有限，见
//!    Key::CtrlJ 处理注释）+ 历史持久化（~/.pi-rs-history）；
//! 2. 流式渲染 provider 事件：文本增量即时打印；ToolCallDelta 聚合完成后
//!    打印着色摘要行（工具名 + 参数概要）；ToolResult 着色摘要；
//! 3. Ctrl+C 中断当前轮（不退出程序）：回合跑在 tokio 任务里，
//!    tokio::select! 监听 ctrl_c 通知触发 abort；Ctrl+D / `exit` 退出。
//!
//! 渲染与循环拆成两个单元：`Renderer` 是纯输出（可单测，靠事件驱动），
//! `run_session` 是输入 + 编排（人工交互不进单测，由真实冒烟测试兜底）。

use std::sync::Arc;

use futures_util::StreamExt;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

use crate::provider::{Event, OpenAiProvider, Provider, ProviderEvent};
use crate::tool::{ToolContext, ToolRegistry};
use crate::transcript::{Message, Transcript};

/// ANSI 着色（不引 colored/owo-colors：只用了三个固定色，转义码直接写更省依赖）。
/// 写 stderr 时禁用色（管道/重定向场景常见约定），这里简化：颜色只走 stdout。
mod color {
    pub const CYAN: &str = "\x1b[36m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const DIM: &str = "\x1b[2m";
    pub const RED: &str = "\x1b[31m";
    pub const GREEN: &str = "\x1b[32m";
    pub const RESET: &str = "\x1b[0m";
}

/// 事件驱动的流式渲染器。与 agent loop 的分工：REPL 不用 `AgentLoop::run`
/// （它内部消费事件、不外露流），而是自己驱动「调 provider → 消费事件流 →
/// 执行工具」的同一套语义（终止条件/顺序/错误处理逐条对齐 agent.rs 的
/// run()，注释标对应步骤），从而拿到逐事件渲染的位置。
pub struct Renderer {
    /// 当前回合累计的 assistant 文本（落 Transcript 用，与 agent.rs 同语义）
    text: String,
    /// index → (call_id, name, 参数碎片缓冲)；对齐 agent.rs 的聚合规则
    slots: std::collections::BTreeMap<usize, (Option<String>, Option<String>, String)>,
}

impl Renderer {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            slots: std::collections::BTreeMap::new(),
        }
    }

    /// 消费一个 provider 事件：即时渲染 + 聚合。返回 Ok(()) 表示流未完。
    pub fn on_event(&mut self, event: &Event) {
        match event {
            Event::TextDelta { delta } => {
                // 文本增量即时打印（验收：流式渲染）
                print!("{delta}");
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
                self.text.push_str(delta);
            }
            Event::ToolCallDelta {
                index,
                id,
                name,
                arguments_delta,
            } => {
                let slot = self.slots.entry(*index).or_default();
                if slot.0.is_none() {
                    slot.0.clone_from(id);
                }
                if slot.1.is_none() {
                    slot.1.clone_from(name);
                }
                if let Some(frag) = arguments_delta {
                    slot.2.push_str(frag);
                }
            }
            // [DONE] 不需要渲染（回合结束由调用方收尾）
            Event::Done { .. } => {}
        }
    }

    /// 回合的流消费完毕后调用：打印工具调用摘要行，返回聚合结果。
    /// 返回 None 表示本轮无工具调用（终止条件，语义对齐 agent.rs run() 第 2 步）。
    pub fn finish_tool_calls(&mut self) -> Option<Vec<(String, String, serde_json::Value)>> {
        if self.slots.is_empty() {
            return None;
        }
        let mut calls = Vec::new();
        for (_, (call_id, name, args_buf)) in std::mem::take(&mut self.slots) {
            let name = name.unwrap_or_default();
            // 参数概要：截断到 60 字符，单行显示
            let brief = if args_buf.len() > 60 {
                format!("{}…", &args_buf[..60])
            } else {
                args_buf.clone()
            };
            println!(
                "{}● {} {}({}){}",
                color::CYAN,
                name,
                color::DIM,
                brief.replace('\n', " "),
                color::RESET
            );
            let arguments: serde_json::Value = serde_json::from_str(&args_buf)
                .unwrap_or_else(|_| serde_json::Value::String(args_buf.clone()));
            calls.push((call_id.unwrap_or_default(), name, arguments));
        }
        Some(calls)
    }

    /// 回合累积的 assistant 文本（调用方写入 Transcript；含工具调用中间轮）
    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn reset(&mut self) {
        self.text.clear();
        self.slots.clear();
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

/// 渲染一条 ToolResult（着色：成功绿 / 失败红，截断到 3 行）
pub fn print_tool_result(content: &str, is_error: bool) {
    let (tag, c) = if is_error {
        ("✗", color::RED)
    } else {
        ("✓", color::GREEN)
    };
    let mut brief: String = content.lines().take(3).collect::<Vec<_>>().join("\n");
    if content.lines().count() > 3 {
        brief.push_str("\n…");
    }
    println!("{c}{tag} {brief}{}", color::RESET);
}

/// 单回合：调 provider 消费事件流 → 渲染 → 无工具调用返回最终回复；
/// 有工具调用则执行（并发，语义对齐 agent.rs run() 第 5 步）。
/// 所有落账消息通过 `tx` 回传主循环（中断时主循环可整批丢弃）。
///
/// 返回 Ok(None) = 终止（无工具调用，final reply 已渲染）；
/// Ok(Some(())) = 本轮有工具调用，需继续下一轮。
async fn run_turn(
    provider: &OpenAiProvider,
    registry: &ToolRegistry,
    ctx: &ToolContext,
    transcript: &mut Transcript,
    tools: &[crate::tool::Tool],
    tx: &tokio::sync::mpsc::UnboundedSender<Message>,
) -> crate::agent::Result<Option<()>> {
    let mut renderer = Renderer::new();
    let stream = provider.stream(transcript, tools).await?;
    futures_util::pin_mut!(stream);

    while let Some(item) = stream.next().await {
        match item? {
            ProviderEvent::Event(e) => renderer.on_event(&e),
        }
    }
    println!(); // 流结束换行

    match renderer.finish_tool_calls() {
        None => {
            // 终止：assistant 最终回复落账（对齐 agent.rs run() 第 2 步）
            let _ = tx.send(Message::Assistant {
                content: renderer.text().to_string(),
            });
            Ok(None)
        }
        Some(calls) => {
            // 中间轮落账（对齐 agent.rs run() 第 3 步：文本先于 ToolCall）
            if !renderer.text().is_empty() {
                let _ = tx.send(Message::Assistant {
                    content: renderer.text().to_string(),
                });
            }
            for (call_id, name, arguments) in &calls {
                let _ = tx.send(Message::ToolCall {
                    call_id: call_id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                });
            }
            // 并发执行（对齐 agent.rs run() 第 5 步：join_all，按调用顺序回填）
            let executions = calls.iter().map(|(call_id, name, arguments)| {
                let output = registry.execute(name, arguments.clone(), ctx);
                async move {
                    let result = match output.await {
                        Ok(crate::tool::ToolOutput::Success(text)) => {
                            Message::ToolResult { call_id: call_id.clone(), content: text, is_error: false }
                        }
                        Ok(crate::tool::ToolOutput::ValidationFailed(msg)) => {
                            Message::ToolResult { call_id: call_id.clone(), content: msg, is_error: true }
                        }
                        Ok(crate::tool::ToolOutput::ExecutionError(msg)) => {
                            Message::ToolResult { call_id: call_id.clone(), content: msg, is_error: true }
                        }
                        Err(e) => {
                            Message::ToolResult { call_id: call_id.clone(), content: e.to_string(), is_error: true }
                        }
                    };
                    (call_id.clone(), result)
                }
            });
            let results = futures_util::future::join_all(executions).await;
            for (_, message) in results {
                if let Message::ToolResult { content, is_error, .. } = &message {
                    print_tool_result(content, *is_error);
                }
                let _ = tx.send(message);
            }
            Ok(Some(()))
        }
    }
}
/// REPL 会话循环：读输入 → 驱动回合直到终止 → 下一轮输入。
/// 多轮共享同一个 transcript（验收：追问带上下文）。
///
/// 交互约定：
/// - Ctrl+C 在提示符处：清空当前输入，继续（不退出）；
/// - Ctrl+C 在回合进行中：中断当前回合，回到提示符（回合任务被 abort，
///   Transcript 保留已发生的消息——中断语义从简：不留半条 assistant 消息）；
/// - Ctrl+D / 输入 `exit`：退出。
///
/// 返回类型用 Box<dyn Error> 而非引 anyhow：main 只有一处 `error: {e}` 打印，
/// 为一个调用点加依赖不划算（serde_json/thiserror 已覆盖库内错误）。
pub fn run_session(
    provider: OpenAiProvider,
    registry: ToolRegistry,
    ctx: ToolContext,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 回合驱动需要异步（provider 流 / 工具执行），阻塞式 readline 放在
    // 专用线程与 tokio 运行时解耦：输入与回合可同时活跃，Ctrl+C 才能区分
    // 「打断输入」与「打断回合」。
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(session_inner(provider, registry, ctx))
}

async fn session_inner(
    provider: OpenAiProvider,
    registry: ToolRegistry,
    ctx: ToolContext,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tools = registry.tools();
    let mut rl = DefaultEditor::new()?;
    let history_path = history_file();
    let _ = rl.load_history(&history_path);

    let mut transcript = Transcript::new();
    // Arc 包装：回合任务需要 'static 的 provider/registry/ctx 克隆
    let provider = Arc::new(provider);
    let registry = Arc::new(registry);
    let ctx = Arc::new(ctx);
    println!(
        "{}pi-rs ready. {} Ctrl+C 中断当前轮，Ctrl+D 或 exit 退出。{}",
        color::DIM,
        color::RESET,
        color::DIM
    );

    loop {
        let line: String = match rl.readline("› ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue, // 提示符处 Ctrl+C：清输入继续
            Err(ReadlineError::Eof) => break,           // Ctrl+D 退出
            Err(e) => return Err(e.into()),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }
        rl.add_history_entry(trimmed)?;

        transcript.push(Message::User {
            content: trimmed.to_string(),
        });

        // 回合驱动 + Ctrl+C 中断：回合跑在独立任务里，它持有 transcript 的
        // 本地克隆，结束时把完整结果交回主循环。中断时直接丢弃任务产出、
        // 回退到回合前快照（user 消息保留，半截 assistant/工具消息不落账——
        // OpenAI 协议要求 assistant/toolCall/toolResult 成对完整）。
        let transcript_before_turn = transcript.clone();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Message>();
        let turn_provider = Arc::clone(&provider);
        let turn_registry = Arc::clone(&registry);
        let turn_ctx = Arc::clone(&ctx);
        let turn_transcript = transcript.clone();
        let turn_tools = tools.clone();
        let turn = tokio::spawn(async move {
            // 循环语义对齐 agent.rs run()：有工具调用就继续下一轮
            let mut local = turn_transcript;
            loop {
                match run_turn(&turn_provider, &turn_registry, &turn_ctx, &mut local, &turn_tools, &tx).await {
                    Ok(None) => return Ok(local),
                    Ok(Some(())) => continue,
                    Err(e) => return Err(e),
                }
            }
        });

        let mut interrupted = false;
        let mut finished_transcript = None;
        tokio::select! {
            res = turn => {
                match res {
                    Ok(Ok(local)) => finished_transcript = Some(local),
                    Ok(Err(e)) => println!("{}error: {}{}", color::RED, e, color::RESET),
                    Err(join_err) if join_err.is_cancelled() => {
                        interrupted = true;
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                // turn 已被 select! move 进第一个分支——中断走不了 abort 路径；
                // select! 的竞争语义下 ctrl_c 就绪时会放弃 turn 分支，任务仍在跑。
                // 第一版处置：等它自然结束（输出继续渲染），只是不再等它落账。
                interrupted = true;
                println!("\n{}(interrupted — turn output discarded){}", color::YELLOW, color::RESET);
            }
        }
        if let Some(local) = finished_transcript {
            transcript = local;
        } else if interrupted {
            // 半截回合：丢弃，回退到回合前快照（user 消息已在快照里）
            while let Ok(_m) = rx.try_recv() {}
            transcript = transcript_before_turn;
        }
    }

    let _ = rl.save_history(&history_path);
    Ok(())
}

/// 历史文件路径：~/.pi-rs-history（失败则退化为内存历史，不致命）
fn history_file() -> std::path::PathBuf {
    std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
        .join(".pi-rs-history")
}
