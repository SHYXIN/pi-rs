//! grep 工具：内容搜索，返回 文件 + 行号 + 行内容（ripgrep 风格输出语义）。
//!
//! 实现选型权衡（ticket 要求明确说明）：
//! - **选了：纯 Rust（regex crate + 手写目录遍历）**。
//!   理由一：`rg` 子进程方案要求目标机装有 ripgrep——Windows 上没有系统包管理器，
//!   用户机器十有八九没有 rg，测试和首跑都会挂，违背「零外部依赖可跑」的交付底线。
//!   理由二：regex crate 本来就在 rust 生态标准位（tokio/serde 同级），为 grep 引入
//!   不增加心智负担。代价：性能远不及 rg（无并行、无 SIMD），但第一版边界是
//!   「agent 能搜到东西」，搜的是 tempfile 级别的小目录，性能不是本票目标。
//! - **没选 walkdir/ignore crate**：手写递归只有 ~30 行，还能精确控制忽略规则
//!   （跳过隐藏目录/二进制文件）；省两个依赖。多目录规模大了再换 ignore crate。

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use regex::Regex;

use super::{Tool, ToolContext, ToolExecutor, ToolOutput};

/// 单条命中：文件路径 + 1-based 行号 + 行内容
struct Match {
    path: PathBuf,
    line_no: usize,
    line_text: String,
}

/// grep 工具执行器。
///
/// 参数：
/// - `pattern`（string, required）：正则表达式（rust regex 语法）
/// - `path`（string, 可选）：搜索起点，目录（递归）或单文件；缺省为 working_dir
///
/// 输出格式（对齐 rg 默认输出）：`相对路径:行号:行内容`，每命中一行一条；
/// 无命中返回提示文本（不是错误——「搜了没有」对模型是有效结果）。
pub struct GrepTool;

impl GrepTool {
    /// 工具声明；schema 与实现同文件演进，避免两处漂移。
    pub fn declaration() -> Tool {
        Tool {
            name: "grep".into(),
            description: "Search files for a regex pattern; returns file:line:text matches".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression to search for"},
                    "path": {"type": "string", "description": "File or directory to search (default: working directory)"}
                },
                "required": ["pattern"]
            }),
        }
    }
}

/// 判定一个路径是否要跳过：隐藏项（`.` 开头）与常见构建产物目录。
/// 目的：别把 target/、.git/ 里几万个产物行灌回模型上下文。
fn should_skip(name: &std::ffi::OsStr) -> bool {
    let name = name.to_string_lossy();
    name.starts_with('.') || name == "target" || name == "node_modules"
}

/// 递归收集待搜文件（同步实现，见 execute 里的 spawn_blocking 说明）。
/// 二进制文件按「含 NUL 字节」启发式跳过——grep 语义面向文本。
fn collect_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return, // 无权限/竞态删除的子目录：跳过而非整体失败
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if should_skip(entry.file_name().as_os_str()) {
            continue;
        }
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => collect_files(&path, out),
            Ok(_) => out.push(path),
            Err(_) => continue,
        }
    }
}

/// 搜单个文件，逐行匹配。读失败（二进制/无权限）按跳过处理。
fn search_file(path: &Path, re: &Regex) -> Vec<Match> {
    let content = match std::fs::read(path) {
        Ok(bytes) => {
            // NUL 字节启发式：大概率二进制，文本化只会产生乱码行
            if bytes.contains(&0u8) {
                return Vec::new();
            }
            match String::from_utf8(bytes) {
                Ok(s) => s,
                Err(_) => return Vec::new(),
            }
        }
        Err(_) => return Vec::new(),
    };
    content
        .lines()
        .enumerate()
        .filter(|(_, line)| re.is_match(line))
        .map(|(i, line)| Match {
            path: path.to_path_buf(),
            line_no: i + 1, // 1-based，对齐 rg / pi 语义
            line_text: line.to_string(),
        })
        .collect()
}

#[async_trait]
impl ToolExecutor for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput {
        let pattern = args
            .get("pattern")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        if pattern.is_empty() {
            return ToolOutput::ExecutionError(
                "grep tool requires a non-empty `pattern` argument (string)".to_string(),
            );
        }

        let re = match Regex::new(pattern) {
            Ok(re) => re,
            Err(e) => {
                return ToolOutput::ExecutionError(format!(
                    "invalid regex `{pattern}`: {e} (fix the pattern and retry)"
                ))
            }
        };

        let root = match args.get("path").and_then(|v| v.as_str()) {
            Some(p) if !p.is_empty() => ctx.working_dir.join(p),
            _ => ctx.working_dir.clone(),
        };

        if !root.exists() {
            return ToolOutput::ExecutionError(format!(
                "search path not found: {} (pass an existing file or directory as `path`)",
                root.display()
            ));
        }

        // 遍历 + 读文件是阻塞 IO；tokio 环境（ADR-0001）下用 spawn_blocking
        // 丢进阻塞线程池，不占异步执行线程。同步段（收集 + 逐文件匹配）整体搬走。
        let handle_root = root.clone();
        let matches = tokio::task::spawn_blocking(move || {
            let mut files = Vec::new();
            if handle_root.is_dir() {
                collect_files(&handle_root, &mut files);
            } else {
                files.push(handle_root.clone());
            }
            let mut matches = Vec::new();
            for f in &files {
                matches.extend(search_file(f, &re));
            }
            matches
        })
        .await;

        let matches = match matches {
            Ok(m) => m,
            Err(e) => return ToolOutput::ExecutionError(format!("search task failed: {e}")),
        };

        if matches.is_empty() {
            return ToolOutput::Success(format!(
                "no matches for pattern `{pattern}`"
            ));
        }

        // 输出路径相对化：working_dir 前缀剥掉，回传模型的文本更短更可读；
        // 分隔符统一为 /（rg 风格），Windows 反斜杠会让输出跨平台不一致
        let mut lines: Vec<String> = matches
            .into_iter()
            .map(|m| {
                let rel = m.path.strip_prefix(&ctx.working_dir).unwrap_or(&m.path);
                let rel = rel.to_string_lossy().replace('\\', "/");
                format!("{}:{}:{}", rel, m.line_no, m.line_text)
            })
            .collect();
        lines.sort(); // 输出顺序稳定（遍历序跨平台不定），测试与模型阅读都受益
        lines.dedup();

        ToolOutput::Success(lines.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// tempfile 沙箱：预置多文件 fixture（对齐 read.rs 的 helper 风格）
    fn sandbox() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.rs"), "fn main() {}\nlet x = 1;\n").unwrap();
        std::fs::create_dir(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("sub/b.rs"), "let y = 2;\nfn helper() {}\n").unwrap();
        dir
    }

    #[tokio::test]
    async fn finds_matches_across_files_with_line_numbers() {
        let dir = sandbox();
        let out = GrepTool
            .execute(json!({"pattern": "fn \\w+"}), &ToolContext::new(dir.path()))
            .await;
        let text = out.success_text().unwrap();
        // ripgrep 风格 file:line:text
        assert!(text.contains("a.rs:1:fn main() {}"), "{text}");
        assert!(text.contains("sub/b.rs:2:fn helper() {}"), "{text}");
    }

    #[tokio::test]
    async fn searches_single_file_path() {
        let dir = sandbox();
        let out = GrepTool
            .execute(
                json!({"pattern": "let", "path": "a.rs"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        let text = out.success_text().unwrap();
        assert!(text.contains("a.rs:2:let x = 1;"), "{text}");
        assert!(!text.contains("b.rs"), "{text}");
    }

    #[tokio::test]
    async fn no_matches_is_success_with_hint() {
        // 「搜了没有」是有效结果不是工具故障
        let dir = sandbox();
        let out = GrepTool
            .execute(json!({"pattern": "nonexistent_pattern_xyz"}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::Success(text) => assert!(text.contains("no matches"), "{text}"),
            other => panic!("expected Success, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn nonexistent_path_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = GrepTool
            .execute(
                json!({"pattern": "x", "path": "no/such/dir"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("not found"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn invalid_regex_is_execution_error() {
        let dir = sandbox();
        let out = GrepTool
            .execute(json!({"pattern": "(unclosed"}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("invalid regex"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn skips_hidden_and_target_dirs() {
        let dir = sandbox();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        std::fs::write(dir.path().join(".git/config"), "let z = 9;\n").unwrap();
        std::fs::create_dir(dir.path().join("target")).unwrap();
        std::fs::write(dir.path().join("target/out.rs"), "let w = 8;\n").unwrap();
        let out = GrepTool
            .execute(json!({"pattern": "let [zwx]"}), &ToolContext::new(dir.path()))
            .await;
        let text = out.success_text().unwrap();
        assert!(!text.contains(".git"), "{text}");
        assert!(!text.contains("target"), "{text}");
    }

    #[tokio::test]
    async fn skips_binary_files() {
        let dir = tempfile::tempdir().unwrap();
        // NUL 字节 = 二进制启发式
        std::fs::write(dir.path().join("blob.bin"), b"let \x00 binary = 1;\n").unwrap();
        let out = GrepTool
            .execute(json!({"pattern": "let"}), &ToolContext::new(dir.path()))
            .await;
        let text = out.success_text().unwrap();
        assert!(text.contains("no matches"), "{text}");
    }

    #[tokio::test]
    async fn empty_pattern_is_execution_error() {
        let dir = sandbox();
        let out = GrepTool
            .execute(json!({"pattern": ""}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("pattern"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[test]
    fn grep_tool_name_matches_registry_key() {
        assert_eq!(GrepTool.name(), "grep");
    }

    #[test]
    fn declaration_is_well_formed() {
        let d = GrepTool::declaration();
        assert_eq!(d.name, "grep");
        assert!(d.is_well_formed());
    }
}
