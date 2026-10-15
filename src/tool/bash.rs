//! bash 工具：执行 shell 命令，捕获 stdout / stderr / 退出码（语义对齐 pi 同名工具）。
//!
//! shell 选型（平台默认 + `shell` 参数可覆盖）：
//! - Windows 默认 `cmd.exe /C`：Windows 上唯一保证存在的 shell。开发机虽是 Git Bash
//!   环境，但目标机不保证装有 Git，`bash`/`sh` 不能当默认。
//! - Unix 默认 `/bin/sh -c`：POSIX 兜底，不依赖 bash 是否安装。
//! - 覆盖只换可执行文件、不换参数约定（Windows 仍 /C、Unix 仍 -c）。
//!   第一版取舍：完整 shell 矩阵（pwsh -Command、Windows 上 bash.exe 需 -c 等）留给后续 ticket。
//!
//! 输出语义：非零退出码是「执行结果」而非「工具故障」——模型需要看到 stderr 与
//! exit code 才能决定下一步怎么修，所以走 Success 并附带退出码；ExecutionError
//! 留给 spawn 失败、超时被杀这类工具级故障。

use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;

use super::{Tool, ToolContext, ToolExecutor, ToolOutput};

/// 默认超时：单条命令 30s 足够，又不至于把 agent loop 长时间挂死
const DEFAULT_TIMEOUT_MS: u64 = 30_000;

/// bash 工具执行器。
///
/// 参数：
/// - `command`（string, required）：要执行的 shell 命令
/// - `timeout_ms`（integer, 可选）：超时毫秒数，超时杀进程（默认 30000）
/// - `shell`（string, 可选）：覆盖 shell 可执行文件（参数约定见模块注释）
pub struct BashTool;

impl BashTool {
    /// 工具声明（name / description / 参数 schema），schema 与实现同文件演进，避免两处漂移。
    pub fn declaration() -> Tool {
        Tool {
            name: "bash".into(),
            description: "Run a shell command and return its output".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to run"},
                    "timeout_ms": {"type": "integer", "description": "Kill the command after this many milliseconds (default 30000)"},
                    "shell": {"type": "string", "description": "Override the shell executable (Windows default: cmd, Unix default: /bin/sh)"}
                },
                "required": ["command"]
            }),
        }
    }
}

/// 平台 shell 调用形状：可执行文件 + 前置参数。
/// Windows: `<shell> /C <command>`；Unix: `<shell> -c <command>`。
fn shell_invocation(command: &str, shell_override: Option<&str>) -> (String, Vec<String>) {
    #[cfg(windows)]
    {
        let program = shell_override.unwrap_or("cmd").to_string();
        (program, vec!["/C".to_string(), command.to_string()])
    }
    #[cfg(not(windows))]
    {
        let program = shell_override.unwrap_or("/bin/sh").to_string();
        (program, vec!["-c".to_string(), command.to_string()])
    }
}

#[async_trait]
impl ToolExecutor for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput {
        let command = args
            .get("command")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        // 注册表已做 schema 校验，这里只在绕过注册表直调时兜底（对齐 read/write 的写法）
        if command.is_empty() {
            return ToolOutput::ExecutionError(
                "bash tool requires a non-empty `command` argument (string)".to_string(),
            );
        }
        let timeout_ms = args
            .get("timeout_ms")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_TIMEOUT_MS);
        if timeout_ms == 0 {
            return ToolOutput::ExecutionError(
                "`timeout_ms` must be a positive integer (milliseconds)".to_string(),
            );
        }

        let (program, shell_args) =
            shell_invocation(command, args.get("shell").and_then(|v| v.as_str()));

        let mut cmd = tokio::process::Command::new(&program);
        cmd.args(&shell_args)
            .current_dir(&ctx.working_dir)
            // stdin 关死：防止子进程等输入挂死；stdout/stderr 必须 piped：wait_with_output 的前提
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // 超时路径的杀进程机制：timeout 掉的是包着 Child 的 future，tokio 的 Child
            // 默认 drop 不杀进程，kill_on_drop(true) 保证 future 被 drop（=超时）时
            // 子进程被 kill，不留孤儿
            .kill_on_drop(true);

        let child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => {
                return ToolOutput::ExecutionError(format!("failed to spawn `{program}`: {e}"))
            }
        };

        match tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait_with_output())
            .await
        {
            Ok(Ok(output)) => {
                // 被信号终止时（Unix）没有退出码，用 -1 占位
                let code = output.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                ToolOutput::Success(format!(
                    "exit code: {code}\nstdout:\n{}\nstderr:\n{}",
                    stdout.trim_end(),
                    stderr.trim_end()
                ))
            }
            Ok(Err(e)) => ToolOutput::ExecutionError(format!("failed to run `{command}`: {e}")),
            Err(_) => ToolOutput::ExecutionError(format!(
                "command timed out after {timeout_ms}ms and was killed: {command}"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// 平台差异收敛在一个 helper：跨平台的「睡约 N 秒」命令（超时测试用）
    fn sleep_command(seconds: u32) -> String {
        if cfg!(windows) {
            // ping 每秒一次，-n k 实际约 k-1 秒；>nul 保持输出干净
            format!("ping -n {} 127.0.0.1 >nul", seconds + 1)
        } else {
            format!("sleep {seconds}")
        }
    }

    #[tokio::test]
    async fn runs_command_and_captures_stdout() {
        let dir = tempfile::tempdir().unwrap();
        let out = BashTool
            .execute(json!({"command": "echo hello"}), &ToolContext::new(dir.path()))
            .await;
        assert!(out.is_success(), "{out:?}");
        let text = out.success_text().unwrap();
        assert!(text.contains("hello"), "{text}");
        assert!(text.contains("exit code: 0"), "{text}");
    }

    #[tokio::test]
    async fn nonzero_exit_code_is_success_with_code() {
        // 非零退出码是执行结果不是工具故障：exit code 进 Success 文本回传模型
        let dir = tempfile::tempdir().unwrap();
        let out = BashTool
            .execute(json!({"command": "exit 3"}), &ToolContext::new(dir.path()))
            .await;
        let text = out.success_text().unwrap();
        assert!(text.contains("exit code: 3"), "{text}");
    }

    #[tokio::test]
    async fn captures_stderr_separately() {
        let dir = tempfile::tempdir().unwrap();
        // `1>&2` 在 cmd 与 sh 里语法一致，都是把 stdout 重定向到 stderr
        let out = BashTool
            .execute(
                json!({"command": "echo oops 1>&2"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        let text = out.success_text().unwrap();
        assert!(text.contains("stderr:"), "{text}");
        assert!(text.contains("oops"), "{text}");
    }

    #[tokio::test]
    async fn timeout_kills_long_command() {
        let dir = tempfile::tempdir().unwrap();
        let out = BashTool
            .execute(
                json!({"command": sleep_command(20), "timeout_ms": 300}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("timed out"), "{msg}");
                assert!(msg.contains("300"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn runs_in_working_dir() {
        // 命令的工作目录是 ctx.working_dir：能直接看到沙箱里的文件
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("marker.txt"), "x").unwrap();
        let cmd_str = if cfg!(windows) { "dir /b" } else { "ls" };
        let out = BashTool
            .execute(json!({"command": cmd_str}), &ToolContext::new(dir.path()))
            .await;
        let text = out.success_text().unwrap();
        assert!(text.contains("marker.txt"), "{text}");
    }

    #[tokio::test]
    async fn empty_command_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = BashTool
            .execute(json!({"command": ""}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("command"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn zero_timeout_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = BashTool
            .execute(
                json!({"command": "echo hi", "timeout_ms": 0}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("timeout_ms"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[test]
    fn bash_tool_name_matches_registry_key() {
        assert_eq!(BashTool.name(), "bash");
    }

    #[test]
    fn declaration_is_well_formed() {
        let d = BashTool::declaration();
        assert_eq!(d.name, "bash");
        assert!(d.is_well_formed());
    }
}
