//! read 工具：读文件内容（语义对齐 pi 同名工具，参数名对齐 ticket：file_path）。

use async_trait::async_trait;

use super::{Tool, ToolContext, ToolExecutor, ToolOutput};

/// read 工具执行器。
///
/// 参数：
/// - `file_path`（string, required）：要读的文件路径，相对路径以 working_dir 为基准
/// - `offset`（integer, 可选）：从第几行开始读（1-based，对齐 pi read 的行号语义）
/// - `limit`（integer, 可选）：最多读多少行
///
/// 取舍说明：pi 的 read 还有格式化行号前缀、截断超长行等富输出行为；
/// 本 ticket 只交付「内容为字符串」的核心语义，行号截取用最简实现
/// （offset 之前丢弃、按 limit 截取），富格式化留给后续追平 pi 的 ticket。
pub struct ReadTool;

impl ReadTool {
    /// 工具声明；schema 与实现同文件演进，避免两处漂移（对齐 bash/edit/grep）。
    pub fn declaration() -> Tool {
        Tool {
            name: "read".into(),
            description: "Read a file's content from disk".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File path to read"},
                    "offset": {"type": "integer", "description": "Line number to start reading from (1-based)"},
                    "limit": {"type": "integer", "description": "Maximum number of lines to read"}
                },
                "required": ["file_path"]
            }),
        }
    }
}

#[async_trait]
impl ToolExecutor for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput {
        let path_str = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let path = ctx.working_dir.join(path_str);

        // 注册表已做 schema 校验，这里的 unwrap_or_default 只在绕过注册表直调时兜底；
        // 空路径没有意义，直接给出可重试的错误消息。
        if path_str.is_empty() {
            return ToolOutput::ExecutionError(
                "read tool requires a non-empty `file_path` argument (string)".to_string(),
            );
        }

        // tokio 环境（ADR-0001）：IO 用 tokio::fs，避免阻塞执行线程
        let metadata = match tokio::fs::metadata(&path).await {
            Ok(m) => m,
            Err(_) => {
                return ToolOutput::ExecutionError(format!(
                    "file not found: {path_str} (pass an existing file path as `file_path`)"
                ))
            }
        };
        if metadata.is_dir() {
            return ToolOutput::ExecutionError(format!(
                "`{path_str}` is a directory, not a file (read expects a file path)"
            ));
        }

        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolOutput::ExecutionError(format!(
                    "failed to read `{path_str}`: {e} (is it valid UTF-8 text?)"
                ))
            }
        };

        let offset = args.get("offset").and_then(|v| v.as_u64());
        let limit = args.get("limit").and_then(|v| v.as_u64());

        let selected: String = match (offset, limit) {
            (None, None) => content,
            (offset, limit) => {
                // offset 1-based：第 n 行即从行号 n 开始
                let start = offset.and_then(|o| o.checked_sub(1)).unwrap_or(0) as usize;
                content
                    .lines()
                    .skip(start)
                    .take(limit.unwrap_or(u64::MAX) as usize)
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };

        ToolOutput::Success(selected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// tempfile 沙箱：写 fixture，测试结束自动清理
    fn sandbox_with_file(rel: &str, content: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(rel);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(path, content).unwrap();
        dir
    }

    #[tokio::test]
    async fn reads_file_content() {
        let dir = sandbox_with_file("a.txt", "hello\nworld\n");
        let out = ReadTool
            .execute(json!({"file_path": "a.txt"}), &ToolContext::new(dir.path()))
            .await;
        assert_eq!(out.success_text(), Some("hello\nworld\n"));
    }

    #[tokio::test]
    async fn reads_with_offset_and_limit() {
        let dir = sandbox_with_file("lines.txt", "l1\nl2\nl3\nl4\nl5\n");
        let ctx = ToolContext::new(dir.path());
        let out = ReadTool
            .execute(json!({"file_path": "lines.txt", "offset": 2, "limit": 2}), &ctx)
            .await;
        assert_eq!(out.success_text(), Some("l2\nl3"));

        let out = ReadTool
            .execute(json!({"file_path": "lines.txt", "offset": 4}), &ctx)
            .await;
        assert_eq!(out.success_text(), Some("l4\nl5"));

        let out = ReadTool
            .execute(json!({"file_path": "lines.txt", "limit": 2}), &ctx)
            .await;
        assert_eq!(out.success_text(), Some("l1\nl2"));
    }

    #[tokio::test]
    async fn reads_nested_relative_path() {
        let dir = sandbox_with_file("sub/dir/f.txt", "nested");
        let out = ReadTool
            .execute(json!({"file_path": "sub/dir/f.txt"}), &ToolContext::new(dir.path()))
            .await;
        assert_eq!(out.success_text(), Some("nested"));
    }

    #[tokio::test]
    async fn missing_file_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = ReadTool
            .execute(json!({"file_path": "absent.txt"}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("absent.txt"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn directory_path_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let out = ReadTool
            .execute(json!({"file_path": "subdir"}), &ToolContext::new(dir.path()))
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("directory"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_file_reads_to_empty_string() {
        let dir = sandbox_with_file("empty.txt", "");
        let out = ReadTool
            .execute(json!({"file_path": "empty.txt"}), &ToolContext::new(dir.path()))
            .await;
        assert_eq!(out.success_text(), Some(""));
    }

    #[tokio::test]
    async fn absolute_path_outside_working_dir_still_reads() {
        // 设计决定：不做路径沙箱限制——pi 语义是「工作目录为相对路径基准」，
        // 不是「禁止绝对路径」；绝对路径能力留给上层策略（后续 ticket）。
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("outside.txt");
        std::fs::write(&file, "outside").unwrap();
        let other = tempfile::tempdir().unwrap();
        let out = ReadTool
            .execute(
                json!({"file_path": file.to_string_lossy()}),
                &ToolContext::new(other.path()),
            )
            .await;
        assert_eq!(out.success_text(), Some("outside"));
    }

    #[test]
    fn read_tool_name_matches_registry_key() {
        assert_eq!(ReadTool.name(), "read");
    }
}
