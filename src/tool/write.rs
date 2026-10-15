//! write 工具：写文件 + 自动建父目录（语义对齐 pi 同名工具，参数名对齐 ticket）。

use async_trait::async_trait;

use super::{ToolContext, ToolExecutor, ToolOutput};

/// write 工具执行器。
///
/// 参数：
/// - `file_path`（string, required）：目标文件路径，相对路径以 working_dir 为基准
/// - `content`（string, required）：要写入的完整内容（覆盖写，不是追加——对齐 pi write 语义）
///
/// 成功输出写入的字节数（UTF-8）。
pub struct WriteTool;

#[async_trait]
impl ToolExecutor for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput {
        let path_str = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let content = args
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or_default();

        if path_str.is_empty() {
            return ToolOutput::ExecutionError(
                "write tool requires a non-empty `file_path` argument (string)".to_string(),
            );
        }

        let path = ctx.working_dir.join(path_str);

        // 拒绝把目录当文件写：先查已有路径是否是目录
        if let Ok(metadata) = tokio::fs::metadata(&path).await {
            if metadata.is_dir() {
                return ToolOutput::ExecutionError(format!(
                    "`{path_str}` is a directory, not a file (write expects a file path)"
                ));
            }
        }

        // 自动创建父目录（对齐 pi write：模型给 nest/new/file.txt 也能写）
        if let Some(parent) = path.parent() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return ToolOutput::ExecutionError(format!(
                    "failed to create parent directories for `{path_str}`: {e}"
                ));
            }
        }

        // tokio 环境（ADR-0001）：IO 用 tokio::fs
        match tokio::fs::write(&path, content).await {
            Ok(()) => ToolOutput::Success(format!("wrote {} bytes to {path_str}", content.len())),
            Err(e) => ToolOutput::ExecutionError(format!("failed to write `{path_str}`: {e}")),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn writes_file_and_reports_byte_count() {
        let dir = tempfile::tempdir().unwrap();
        let out = WriteTool
            .execute(
                json!({"file_path": "a.txt", "content": "hello"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        assert_eq!(
            out,
            ToolOutput::Success("wrote 5 bytes to a.txt".to_string())
        );
        assert_eq!(std::fs::read_to_string(dir.path().join("a.txt")).unwrap(), "hello");
    }

    #[tokio::test]
    async fn creates_parent_directories() {
        let dir = tempfile::tempdir().unwrap();
        let out = WriteTool
            .execute(
                json!({"file_path": "deep/nest/ed/f.txt", "content": "x"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        assert!(out.is_success(), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("deep/nest/ed/f.txt")).unwrap(),
            "x"
        );
    }

    #[tokio::test]
    async fn overwrites_existing_file() {
        // 对齐 pi：write 是覆盖写，不是追加
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "old content").unwrap();
        let ctx = ToolContext::new(dir.path());
        let out = WriteTool
            .execute(json!({"file_path": "a.txt", "content": "new"}), &ctx)
            .await;
        assert!(out.is_success(), "{out:?}");
        assert_eq!(std::fs::read_to_string(dir.path().join("a.txt")).unwrap(), "new");
    }

    #[tokio::test]
    async fn empty_content_writes_empty_file() {
        let dir = tempfile::tempdir().unwrap();
        let out = WriteTool
            .execute(json!({"file_path": "e.txt", "content": ""}), &ToolContext::new(dir.path()))
            .await;
        assert_eq!(out, ToolOutput::Success("wrote 0 bytes to e.txt".to_string()));
        assert_eq!(std::fs::read_to_string(dir.path().join("e.txt")).unwrap(), "");
    }

    #[tokio::test]
    async fn directory_target_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let out = WriteTool
            .execute(
                json!({"file_path": "subdir", "content": "x"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("directory"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[test]
    fn write_tool_name_matches_registry_key() {
        assert_eq!(WriteTool.name(), "write");
    }
}
