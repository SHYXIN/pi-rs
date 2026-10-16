//! REPL：终端入口（GLOSSARY.md First Version Boundary 的最后一环）。
//!
//! 职责：
//! 1. rustyline 输入（Alt+Enter / Ctrl-J 换行续写，见下方多行输入说明）
//!    + 历史持久化（~/.pi-rs-history）；
//! 2. 流式渲染：文本增量即时打印；工具调用打印着色摘要行（工具名 + 参数
//!    概要）；工具结果打印 ✓/✗ 摘要。渲染挂在 `AgentLoop::run_with_events`
//!    的观察者回调上（hardening：此前 REPL 自己复刻一份循环语义，与
//!    agent.rs 双份维护；现在循环/聚合/错误处理单一来源在 agent.rs，
//!    MAX_ITERATIONS 上限与 beforeToolCall hook seam 也随之在 REPL 生效）；
//! 3. Ctrl+C 真中断当前轮（不退出程序）：回合跑在 tokio 任务里，
//!    tokio::select! 监听 ctrl_c 触发 `turn.abort()`（provider 流 / 工具
//!    future 被 drop，bash 子进程由 kill_on_drop 收尾），Transcript 回退到
//!    回合前快照；Ctrl+D / `exit` 退出。
//!
//! 多行输入说明（诚实降级）：多数终端把 Shift+Enter 与 Enter 发出同一个
//! 字节，rustyline 无法区分；真正的 Shift+Enter 需 kitty keyboard protocol
//! 终端支持，第一版不做。此处绑定 Alt+Enter 为换行（Ctrl-J rustyline 原生
//! 支持），Enter 提交。
//!
//! 渲染与循环拆成两个单元：`Renderer` 是纯输出（可单测，靠 AgentEvent 驱动），
//! `run_session` 是输入 + 编排（人工交互不进单测，由真实冒烟测试兜底）。

use std::sync::Arc;

use rustyline::error::ReadlineError;
use rustyline::history::FileHistory;
use rustyline::{Cmd, Editor, KeyCode, KeyEvent};

use crate::agent::{AgentEvent, AgentLoop, BeforeToolCallHook};
use crate::provider::OpenAiProvider;
use crate::tool::{ToolContext, ToolRegistry};
use crate::transcript::Transcript;

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

/// 事件驱动的流式渲染器：纯输出，只消费 `AgentEvent`（循环语义全部在
/// agent.rs 的 `AgentLoop`，本结构不再聚合/解析任何协议碎片）。
pub struct Renderer;

impl Renderer {
    pub fn new() -> Self {
        Self
    }

    /// 渲染一个 agent loop 事件：文本增量即时打印；工具调用打印着色摘要
    /// 行；工具结果打印 ✓/✗ 摘要（截断到 3 行）。
    pub fn on_agent_event(&mut self, event: AgentEvent<'_>) {
        match event {
            AgentEvent::Stream(crate::provider::Event::TextDelta { delta }) => {
                // 文本增量即时打印（验收：流式渲染）
                print!("{delta}");
                use std::io::Write as _;
                let _ = std::io::stdout().flush();
            }
            AgentEvent::ToolCallStarted { name, arguments } => {
                println!(
                    "{}● {} {}({}){}",
                    color::CYAN,
                    name,
                    color::DIM,
                    truncate_brief(&arguments.to_string(), 60).replace('\n', " "),
                    color::RESET
                );
            }
            AgentEvent::ToolResult {
                content, is_error, ..
            } => {
                print_tool_result(content, is_error);
            }
            // 完成信号与工具调用碎片不需要渲染（聚合在 agent.rs，摘要行
            // 由 ToolCallStarted 触发一次，不逐 delta 打印）
            _ => {}
        }
    }
}

impl Default for Renderer {
    fn default() -> Self {
        Self::new()
    }
}

/// 字节安全截断：先按字节上限截，再回退到最近 char boundary（hardening 修复：
/// 原实现 `&s[..60]` 遇到多字节 UTF-8（如中文参数）会 panic）。
fn truncate_brief(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}…", &s[..cut])
}

/// 渲染一条 ToolResult（着色：成功绿 / 失败红，截断到 3 行）
fn print_tool_result(content: &str, is_error: bool) {
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

/// REPL 会话循环：读输入 → 驱动回合直到终止 → 下一轮输入。
/// 多轮共享同一个 transcript（验收：追问带上下文）。
///
/// 交互约定：
/// - Ctrl+C 在提示符处：清空当前输入，继续（不退出）；
/// - Ctrl+C 在回合进行中：abort 回合任务（真取消，非等它跑完），
///   Transcript 回退到回合前快照，回到提示符；
/// - Ctrl+D / 输入 `exit`：退出。
///
/// 返回类型用 Box<dyn Error> 而非引 anyhow：main 只有一处 `error: {e}` 打印，
/// 为一个调用点加依赖不划算（serde_json/thiserror 已覆盖库内错误）。
pub fn run_session(
    provider: Arc<OpenAiProvider>,
    registry: Arc<ToolRegistry>,
    ctx: ToolContext,
    before_tool_call: Option<BeforeToolCallHook>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // 回合驱动需要异步（provider 流 / 工具执行），阻塞式 readline 在
    // block_on 的主线程上：输入与回合串行交替（REPL 的自然节奏），
    // 回合内的 Ctrl+C 由 select! 的 ctrl_c 分支处理（见下）。
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build()?;
    rt.block_on(session_inner(provider, registry, ctx, before_tool_call))
}

async fn session_inner(
    provider: Arc<OpenAiProvider>,
    registry: Arc<ToolRegistry>,
    ctx: ToolContext,
    before_tool_call: Option<BeforeToolCallHook>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut rl = Editor::<(), FileHistory>::new()?;
    // Alt+Enter 换行（Shift+Enter 与 Enter 同字节，见模块文档的诚实降级说明）
    rl.bind_sequence(KeyEvent(KeyCode::Enter, rustyline::Modifiers::ALT), Cmd::Newline);
    let history_path = history_file();
    let _ = rl.load_history(&history_path);

    let mut transcript = Transcript::new();
    println!(
        "{}pi-rs ready. {}Alt+Enter 换行；Ctrl+C 中断当前轮；Ctrl+D 或 exit 退出。{}",
        color::DIM,
        color::RESET,
        color::DIM
    );

    loop {
        let line: String = match rl.readline("› ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue, // 提示符处 Ctrl+C：清输入继续
            Err(ReadlineError::Eof) => break,           // Ctrl+D 退出
            Err(e) => return Err(Box::new(e)),
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "exit" || trimmed == "quit" {
            break;
        }
        rl.add_history_entry(trimmed)?;

        transcript.push(crate::transcript::Message::User {
            content: trimmed.to_string(),
        });

        // 回合任务持有 AgentLoop（Arc 字段跨越 'static 边界）与 transcript
        // 本地克隆；结束时把完整结果交回主循环。中断时 abort 任务并把
        // transcript 回退到回合前快照（半截 assistant/工具消息不落账——
        // OpenAI 协议要求 assistant/toolCall/toolResult 成对完整）。
        let transcript_before_turn = transcript.clone();
        let turn_provider = Arc::clone(&provider);
        let turn_registry = Arc::clone(&registry);
        let turn_ctx = ctx.clone();
        let turn_hook = before_tool_call.clone();
        let turn_transcript = transcript.clone();
        let turn = tokio::spawn(async move {
            let mut local = turn_transcript;
            let mut renderer = Renderer::new();
            let loop_ = AgentLoop::new(turn_provider, turn_registry, turn_ctx);
            let loop_ = if let Some(hook) = turn_hook {
                loop_.with_before_tool_call(hook)
            } else {
                loop_
            };
            loop_
                .run_with_events(&mut local, &mut |ev| renderer.on_agent_event(ev))
                .await
                .map(|_| local)
        });

        // JoinHandle 以借用参与 select（不 move 进分支），ctrl_c 分支才能
        // turn.abort()（hardening 修复：原先 turn 被 move 进第一分支，
        // 中断路径无法取消任务，只能等它跑完并丢弃输出）。
        let mut turn = turn;
        tokio::select! {
            res = &mut turn => {
                match res {
                    Ok(Ok(local)) => transcript = local,
                    Ok(Err(e)) => println!("{}error: {}{}", color::RED, e, color::RESET),
                    Err(e) => return Err(e.into()),
                }
            }
            _ = tokio::signal::ctrl_c() => {
                turn.abort();
                println!("\n{}(interrupted — turn aborted){}", color::YELLOW, color::RESET);
                transcript = transcript_before_turn;
            }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_brief_ascii_short_passes_through() {
        assert_eq!(truncate_brief("hello", 60), "hello");
    }

    #[test]
    fn truncate_brief_ascii_long_cuts_at_limit() {
        let s = "a".repeat(100);
        let out = truncate_brief(&s, 60);
        assert!(out.starts_with(&"a".repeat(60)));
        assert!(out.ends_with('…'));
        assert_eq!(out.chars().count(), 61);
    }

    #[test]
    fn truncate_brief_exactly_at_limit_no_ellipsis() {
        let s = "b".repeat(60);
        assert_eq!(truncate_brief(&s, 60), s);
    }

    #[test]
    fn truncate_brief_multibyte_does_not_panic_and_cuts_at_boundary() {
        // 每个汉字 3 字节：30 字 = 90 字节 > 60；60 不是 char boundary
        let s = "汉".repeat(30);
        let out = truncate_brief(&s, 60);
        assert!(out.ends_with('…'));
        // 60/3=20 字整除边界：截出 20 个完整汉字
        assert_eq!(out.chars().count(), 21);
        assert!(out.starts_with(&"汉".repeat(20)));
    }

    #[test]
    fn truncate_brief_mixed_multibyte_falls_back_to_boundary() {
        // "汉a汉a..."：截到 60 字节会切进汉字（每单元 3+1 字节），回退到边界
        let s = "汉a".repeat(40); // 160 bytes
        let out = truncate_brief(&s, 60);
        assert!(out.ends_with('…'));
        let stem: String = out.trim_end_matches('…').to_string();
        assert!(stem.len() <= 60);
        assert!(stem.is_char_boundary(stem.len()));
    }
}
