//! edit 工具：精确字符串替换（语义对齐 pi 同名工具，参数名对齐 ticket：file_path）。
//!
//! 核心约束：old 串必须在文件中恰好匹配一次——
//! - 零匹配：报错，不落盘
//! - 多匹配：报错，不落盘（「要改哪一处」有歧义，替模型改第一处是危险猜测）
//!
//! 唯一匹配才写盘，且「先全量读、内存替换、再写回」，读写之间不持锁——
//! 单进程 agent 场景下没有并发写同一文件的对手方，锁是过度设计。

use async_trait::async_trait;

use super::{Tool, ToolContext, ToolExecutor, ToolOutput};

/// edit 工具执行器。
///
/// 参数：
/// - `file_path`（string, required）：目标文件路径，相对路径以 working_dir 为基准
/// - `old`（string, required）：被替换的精确文本（唯一匹配才执行）
/// - `new`（string, required）：替换后的文本
pub struct EditTool;

impl EditTool {
    /// 工具声明；schema 与实现同文件演进，避免两处漂移。
    pub fn declaration() -> Tool {
        Tool {
            name: "edit".into(),
            description: "Edit a file by replacing an exact string; the old string must appear exactly once".into(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "File to edit"},
                    "old": {"type": "string", "description": "Exact text to replace (must match exactly once)"},
                    "new": {"type": "string", "description": "Replacement text"}
                },
                "required": ["file_path", "old", "new"]
            }),
        }
    }
}

#[async_trait]
impl ToolExecutor for EditTool {
    fn name(&self) -> &str {
        "edit"
    }

    async fn execute(&self, args: serde_json::Value, ctx: &ToolContext) -> ToolOutput {
        let path_str = args
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        let old = args.get("old").and_then(|v| v.as_str()).unwrap_or_default();
        let new = args.get("new").and_then(|v| v.as_str()).unwrap_or_default();

        // 兜底（对齐 read/write）：注册表已校验 required，这里防绕过注册表直调
        if path_str.is_empty() {
            return ToolOutput::ExecutionError(
                "edit tool requires a non-empty `file_path` argument (string)".to_string(),
            );
        }
        if old.is_empty() {
            return ToolOutput::ExecutionError(
                "edit tool requires a non-empty `old` argument (empty string would match everywhere)"
                    .to_string(),
            );
        }

        let path = ctx.working_dir.join(path_str);

        // tokio 环境（ADR-0001）：IO 用 tokio::fs
        let content = match tokio::fs::read_to_string(&path).await {
            Ok(c) => c,
            Err(e) => {
                return ToolOutput::ExecutionError(format!("failed to read `{path_str}`: {e}"))
            }
        };

        match content.matches(old).count() {
            0 => ToolOutput::ExecutionError(format!(
                "old string not found in `{path_str}`; no changes written (read the file first to get the exact text)"
            )),
            n if n > 1 => ToolOutput::ExecutionError(format!(
                "old string matches {n} times in `{path_str}`; no changes written (include more surrounding text so the match is unique)"
            )),
            1 => {
                let updated = content.replacen(old, new, 1);
                match tokio::fs::write(&path, updated).await {
                    Ok(()) => ToolOutput::Success(format!("edited {path_str}")),
                    Err(e) => ToolOutput::ExecutionError(format!(
                        "failed to write `{path_str}`: {e}"
                    )),
                }
            }
            _ => unreachable!("matches().count() is 0, 1, or more"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// tempfile 沙箱：写 fixture，测试结束自动清理（对齐 read.rs 的 helper）
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
    async fn replaces_unique_match() {
        let dir = sandbox_with_file("a.txt", "hello world\n");
        let out = EditTool
            .execute(
                json!({"file_path": "a.txt", "old": "world", "new": "rust"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        assert!(out.is_success(), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "hello rust\n"
        );
    }

    #[tokio::test]
    async fn zero_matches_is_error_without_write() {
        let dir = sandbox_with_file("a.txt", "untouched\n");
        let out = EditTool
            .execute(
                json!({"file_path": "a.txt", "old": "absent", "new": "x"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("not found"), "{msg}");
                assert!(msg.contains("no changes written"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
        // 关键验收：报错且不落盘
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "untouched\n"
        );
    }

    #[tokio::test]
    async fn multiple_matches_is_error_without_write() {
        let dir = sandbox_with_file("a.txt", "dup dup dup\n");
        let out = EditTool
            .execute(
                json!({"file_path": "a.txt", "old": "dup", "new": "x"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => {
                assert!(msg.contains("3 times"), "{msg}");
                assert!(msg.contains("no changes written"), "{msg}");
            }
            other => panic!("expected ExecutionError, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "dup dup dup\n"
        );
    }

    #[tokio::test]
    async fn multiline_replacement() {
        // old/new 带换行也能精确匹配（模型编辑代码的典型场景）
        let dir = sandbox_with_file("code.rs", "fn a() {\n    todo!()\n}\n");
        let out = EditTool
            .execute(
                json!({"file_path": "code.rs", "old": "    todo!()\n", "new": "    println!(\"hi\");\n"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        assert!(out.is_success(), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("code.rs")).unwrap(),
            "fn a() {\n    println!(\"hi\");\n}\n"
        );
    }

    #[tokio::test]
    async fn missing_file_is_execution_error() {
        let dir = tempfile::tempdir().unwrap();
        let out = EditTool
            .execute(
                json!({"file_path": "absent.txt", "old": "a", "new": "b"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("absent.txt"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn empty_old_is_execution_error() {
        let dir = sandbox_with_file("a.txt", "content\n");
        let out = EditTool
            .execute(
                json!({"file_path": "a.txt", "old": "", "new": "x"}),
                &ToolContext::new(dir.path()),
            )
            .await;
        match out {
            ToolOutput::ExecutionError(msg) => assert!(msg.contains("old"), "{msg}"),
            other => panic!("expected ExecutionError, got {other:?}"),
        }
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "content\n"
        );
    }

    #[tokio::test]
    async fn empty_new_deletes_text() {
        // new 为空是合法的删除操作，区别于 old 为空的参数错误
        let dir = sandbox_with_file("a.txt", "keep DELETE keep\n");
        let out = EditTool
            .execute(
                json!({"file_path": "a.txt", "old": "DELETE ", "new": ""}),
                &ToolContext::new(dir.path()),
            )
            .await;
        assert!(out.is_success(), "{out:?}");
        assert_eq!(
            std::fs::read_to_string(dir.path().join("a.txt")).unwrap(),
            "keep keep\n"
        );
    }

    #[test]
    fn edit_tool_name_matches_registry_key() {
        assert_eq!(EditTool.name(), "edit");
    }

    #[test]
    fn declaration_is_well_formed() {
        let d = EditTool::declaration();
        assert_eq!(d.name, "edit");
        assert!(d.is_well_formed());
    }
}
