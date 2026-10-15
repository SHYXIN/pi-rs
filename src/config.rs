//! config：从本机配置装载 LLM 连接三元组（endpoint / api key / model）。
//!
//! 来源与密钥安全约定见 docs/model-config.md 与 AGENTS.md：
//! 配置文件默认为 `~/.codebuddy/models.json`（本机已有，含真实 key，不入库），
//! 可用环境变量 `PIRS_CONFIG` 覆盖路径。测试一律用 tempfile fixture，不读真实文件。

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::tool::Tool;

/// 默认模型 id（spec 指定）
pub const DEFAULT_MODEL: &str = "glm-5.3-flash";

/// 环境变量名：覆盖配置文件路径
pub const CONFIG_ENV_VAR: &str = "PIRS_CONFIG";

/// models.json 中单个模型条目（只取本 ticket 需要的字段，其余忽略）
#[derive(Debug, Clone, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    #[serde(rename = "apiKey")]
    pub api_key: String,
    pub url: String,
}

/// models.json 顶层结构
#[derive(Debug, Deserialize)]
pub struct ModelsFile {
    pub models: Vec<ModelEntry>,
}

/// 选定模型后的连接三元组。api_key 不实现 Debug/Display，避免日志泄漏。
#[derive(Clone)]
pub struct Connection {
    pub model_id: String,
    pub endpoint: String,
    api_key: String,
}

impl Connection {
    pub fn api_key(&self) -> &str {
        &self.api_key
    }

    /// 测试专用：任意构造连接三元组（密钥安全不受影响——本函数只该在测试里用）
    #[cfg(test)]
    pub fn for_tests() -> Self {
        Self {
            model_id: String::new(),
            endpoint: String::new(),
            api_key: String::new(),
        }
    }

    /// 跨 crate 测试（integration/集成测试）构造：pub 但带 cfg(test) 之外的显式命名，
    /// 供 provider mock 测试使用；api_key 走 setter 保持在 Debug 脱敏路径上。
    #[cfg(any(test, feature = "test-util"))]
    pub fn with_parts(model_id: &str, endpoint: &str, api_key: &str) -> Self {
        Self {
            model_id: model_id.to_string(),
            endpoint: endpoint.to_string(),
            api_key: api_key.to_string(),
        }
    }
}

/// 手写 Debug：只打印 endpoint 与 model，绝不打印 key 明文
impl std::fmt::Debug for Connection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Connection")
            .field("model_id", &self.model_id)
            .field("endpoint", &self.endpoint)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config file not found: {path}")]
    NotFound { path: PathBuf },
    #[error("failed to read config file {path}: {source}")]
    Read {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: serde_json::Error,
    },
    #[error("model `{model_id}` not found in config; available: {available}")]
    ModelNotFound {
        model_id: String,
        available: String,
    },
    #[error("env var {CONFIG_ENV_VAR} is set but empty")]
    EmptyEnvVar,
}

/// 解析配置文件路径：PIRS_CONFIG 环境变量优先，否则 ~/.codebuddy/models.json
pub fn config_path() -> Result<PathBuf, ConfigError> {
    if let Ok(p) = std::env::var(CONFIG_ENV_VAR) {
        if p.trim().is_empty() {
            return Err(ConfigError::EmptyEnvVar);
        }
        return Ok(PathBuf::from(p));
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .map(PathBuf::from)
        .map_err(|_| ConfigError::NotFound {
            path: PathBuf::from("~/.codebuddy/models.json"),
        })?;
    Ok(home.join(".codebuddy").join("models.json"))
}

/// 从给定路径装载 models.json
pub fn load_models_file(path: &Path) -> Result<ModelsFile, ConfigError> {
    if !path.exists() {
        return Err(ConfigError::NotFound {
            path: path.to_path_buf(),
        });
    }
    let raw = std::fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&raw).map_err(|source| ConfigError::Parse {
        path: path.to_path_buf(),
        source,
    })
}

/// 装载配置并按 model_id 选模型（找不到时列出可选 id 帮助排错）
pub fn load_connection(path: &Path, model_id: &str) -> Result<Connection, ConfigError> {
    let file = load_models_file(path)?;
    let entry = file
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| ConfigError::ModelNotFound {
            model_id: model_id.to_string(),
            available: {
                let mut ids: Vec<&str> = file.models.iter().map(|m| m.id.as_str()).collect();
                ids.sort_unstable();
                ids.dedup();
                ids.join(", ")
            },
        })?;
    Ok(Connection {
        model_id: entry.id.clone(),
        endpoint: entry.url.clone(),
        api_key: entry.api_key.clone(),
    })
}

/// 第一版工具集声明（GLOSSARY.md First Version Boundary：read/bash/edit/write/grep）。
/// schema 细化在后续 ticket；此处给出满足 function calling 结构的最小声明。
pub fn default_tools() -> Vec<Tool> {
    let object_schema = |props: &[(&str, &str)]| {
        serde_json::json!({
            "type": "object",
            "properties": props.iter().map(|(n, d)| (n.to_string(), serde_json::json!({"type": "string", "description": d}))).collect::<serde_json::Map<String, serde_json::Value>>()
        })
    };
    vec![
        Tool {
            name: "read".into(),
            description: "Read a file's content from disk".into(),
            parameters: object_schema(&[("path", "File path to read")]),
        },
        Tool {
            name: "bash".into(),
            description: "Run a shell command and return its output".into(),
            parameters: object_schema(&[("command", "The shell command to run")]),
        },
        Tool {
            name: "edit".into(),
            description: "Edit a file by string replacement".into(),
            parameters: object_schema(&[("path", "File path"), ("old", "Text to replace"), ("new", "Replacement text")]),
        },
        Tool {
            name: "write".into(),
            description: "Write content to a file".into(),
            parameters: object_schema(&[("path", "File path"), ("content", "Content to write")]),
        },
        Tool {
            name: "grep".into(),
            description: "Search a pattern in files".into(),
            parameters: object_schema(&[("pattern", "Regex pattern"), ("path", "Directory or file to search")]),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用 tempfile 写 fixture，绝不读本机真实 key 文件
    fn write_fixture(json: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("models.json"), json).unwrap();
        dir
    }

    const FIXTURE: &str = r#"{
        "models": [
            {"id": "glm-5.3-flash", "apiKey": "sk-test-fake-key", "url": "https://example.com/v1/chat/completions"},
            {"id": "gpt-test", "apiKey": "sk-test-other", "url": "https://other.example.com/v1"}
        ]
    }"#;

    #[test]
    fn loads_connection_triple_for_existing_model() {
        let dir = write_fixture(FIXTURE);
        let path = dir.path().join("models.json");

        let conn = load_connection(&path, "glm-5.3-flash").unwrap();
        assert_eq!(conn.model_id, "glm-5.3-flash");
        assert_eq!(conn.endpoint, "https://example.com/v1/chat/completions");
        assert_eq!(conn.api_key(), "sk-test-fake-key");
    }

    #[test]
    fn selects_model_by_id_not_first_entry() {
        let dir = write_fixture(FIXTURE);
        let path = dir.path().join("models.json");

        let conn = load_connection(&path, "gpt-test").unwrap();
        assert_eq!(conn.endpoint, "https://other.example.com/v1");
    }

    #[test]
    fn unknown_model_lists_available_ids() {
        let dir = write_fixture(FIXTURE);
        let path = dir.path().join("models.json");

        let err = load_connection(&path, "nope").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("nope"), "message should name requested model");
        assert!(msg.contains("glm-5.3-flash"), "message should list options");
    }

    #[test]
    fn missing_file_is_not_found_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = load_connection(&dir.path().join("absent.json"), "x").unwrap_err();
        assert!(matches!(err, ConfigError::NotFound { .. }));
    }

    #[test]
    fn malformed_json_is_parse_error() {
        let dir = write_fixture("{ not json");
        let err = load_connection(&dir.path().join("models.json"), "x").unwrap_err();
        assert!(matches!(err, ConfigError::Parse { .. }));
    }

    #[test]
    fn debug_output_never_contains_key() {
        let dir = write_fixture(FIXTURE);
        let path = dir.path().join("models.json");

        let conn = load_connection(&path, "glm-5.3-flash").unwrap();
        let dbg = format!("{conn:?}");
        assert!(!dbg.contains("sk-test-fake-key"), "Debug leaked key: {dbg}");
        assert!(dbg.contains("<redacted>"));
    }

    #[test]
    fn default_tools_are_well_formed() {
        let tools = default_tools();
        assert_eq!(tools.len(), 5);
        assert!(tools.iter().all(|t| t.is_well_formed()));
        let names: Vec<&str> = tools.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, vec!["read", "bash", "edit", "write", "grep"]);
    }
}
