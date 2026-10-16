//! 参数 JSON Schema 轻量校验。
//!
//! 覆盖第一版工具集用到的子集：顶层 `type: object`、`required` 字段存在、
//! `properties` 中各字段的类型检查（string / number / boolean / integer /
//! array / object / null 七种 JSON 原生类型）。不支持 $ref、oneOf、嵌套
//! object 深校验等完整 JSON Schema 语义——第一版工具参数都是一层扁平
//! object，够用且行为可预测（注释取舍，实现完整校验是后续演进）。

/// 校验失败原因，带字段名，最终渲染成可回传模型重试的消息。
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ValidationError {
    #[error("invalid arguments for tool `{tool}`: arguments must be a JSON object, got {got}")]
    NotAnObject { tool: String, got: String },
    #[error(
        "invalid arguments for tool `{tool}`: missing required field `{field}` (type: {expected})"
    )]
    MissingRequired {
        tool: String,
        field: String,
        expected: String,
    },
    #[error(
        "invalid arguments for tool `{tool}`: field `{field}` must be of type {expected}, got {got}"
    )]
    WrongType {
        tool: String,
        field: String,
        expected: String,
        got: String,
    },
}

/// schema 中字段声明的类型名。
const KNOWN_TYPES: [&str; 7] = [
    "string", "number", "boolean", "integer", "array", "object", "null",
];

/// 按 schema 校验 args。
///
/// - args 必须是 JSON object（对应顶层 `"type": "object"`）
/// - `required` 数组里的字段必须存在（类型取该字段在 properties 里的声明）
/// - `properties` 里声明的字段若出现，类型必须匹配
///
/// schema 本身畸形（非 object / type 不是 object）返回
/// [`ValidationError::NotAnObject`]——这属于工具声明错误，不是模型参数错，
/// 但消息同样可回传（措辞区分了 arguments 与 schema）。
pub fn validate_args(tool: &str, schema: &serde_json::Value, args: &serde_json::Value) -> Result<(), ValidationError> {
    let obj = args.as_object().ok_or_else(|| ValidationError::NotAnObject {
        tool: tool.to_string(),
        got: json_type_name(args).to_string(),
    })?;
    let _ = obj;

    let schema_obj = match schema.as_object() {
        Some(s) => s,
        None => {
            return Err(ValidationError::NotAnObject {
                tool: tool.to_string(),
                got: json_type_name(schema).to_string(),
            })
        }
    };
    match schema_obj.get("type").and_then(|t| t.as_str()) {
        Some("object") | None => {}
        Some(other) => {
            return Err(ValidationError::NotAnObject {
                tool: tool.to_string(),
                got: other.to_string(),
            })
        }
    }

    let properties = schema_obj.get("properties").and_then(|p| p.as_object());

    // required：字段必须存在
    if let Some(required) = schema_obj.get("required").and_then(|r| r.as_array()) {
        for req in required {
            let field = match req.as_str() {
                Some(f) => f,
                None => continue, // 畸形 schema 的 required 项，跳过
            };
            if !obj.contains_key(field) {
                let expected = properties
                    .and_then(|p| p.get(field))
                    .and_then(|def| def.get("type"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("any")
                    .to_string();
                return Err(ValidationError::MissingRequired {
                    tool: tool.to_string(),
                    field: field.to_string(),
                    expected,
                });
            }
        }
    }

    // properties：出现的字段类型必须匹配
    if let Some(props) = properties {
        for (field, def) in props {
            let expected = match def.get("type").and_then(|t| t.as_str()) {
                Some(t) if KNOWN_TYPES.contains(&t) => t,
                _ => continue, // 未声明类型或声明了不认识的类型，不检查
            };
            if let Some(value) = obj.get(field) {
                if !type_matches(expected, value) {
                    return Err(ValidationError::WrongType {
                        tool: tool.to_string(),
                        field: field.clone(),
                        expected: expected.to_string(),
                        got: json_type_name(value).to_string(),
                    });
                }
            }
        }
    }

    Ok(())
}

/// 值的实际 JSON 类型名（用于错误消息）。
fn json_type_name(value: &serde_json::Value) -> &'static str {
    match value {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// schema 类型声明与实际值是否匹配。
///
/// `integer` 接受整数值 number；`number` 接受一切 number（含整数）。
fn type_matches(expected: &str, value: &serde_json::Value) -> bool {
    match expected {
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        "null" => value.is_null(),
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn read_schema() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "integer"}
            },
            "required": ["path"]
        })
    }

    #[test]
    fn accepts_valid_args() {
        let args = json!({"path": "a.txt", "offset": 3});
        assert!(validate_args("read", &read_schema(), &args).is_ok());
    }

    #[test]
    fn accepts_optional_fields_absent() {
        let args = json!({"path": "a.txt"});
        assert!(validate_args("read", &read_schema(), &args).is_ok());
    }

    // —— 校验失败用例（ticket 要求至少 3 个：缺 required / 类型错 / 非 object）——

    #[test]
    fn rejects_missing_required_field() {
        let args = json!({"offset": 1});
        let err = validate_args("read", &read_schema(), &args).unwrap_err();
        assert!(matches!(err, ValidationError::MissingRequired { .. }));
        let msg = err.to_string();
        assert!(msg.contains("path"), "message should name the field: {msg}");
        assert!(msg.contains("read"), "message should name the tool: {msg}");
    }

    #[test]
    fn rejects_wrong_type() {
        let args = json!({"path": 42});
        let err = validate_args("read", &read_schema(), &args).unwrap_err();
        assert!(matches!(err, ValidationError::WrongType { .. }));
        let msg = err.to_string();
        assert!(msg.contains("path"), "{msg}");
        assert!(msg.contains("string"), "{msg}");
        assert!(msg.contains("number"), "{msg}");
    }

    #[test]
    fn rejects_non_object_args() {
        let args = json!("just a string");
        let err = validate_args("read", &read_schema(), &args).unwrap_err();
        assert!(matches!(err, ValidationError::NotAnObject { .. }));
        assert!(err.to_string().contains("JSON object"));
    }

    #[test]
    fn rejects_non_object_args_array() {
        let args = json!([1, 2, 3]);
        let err = validate_args("read", &read_schema(), &args).unwrap_err();
        assert!(matches!(err, ValidationError::NotAnObject { .. }));
    }

    #[test]
    fn integer_accepts_int_rejects_float() {
        let schema = json!({
            "type": "object",
            "properties": {"offset": {"type": "integer"}},
            "required": []
        });
        assert!(validate_args("read", &schema, &json!({"offset": 5})).is_ok());
        let err = validate_args("read", &schema, &json!({"offset": 1.5})).unwrap_err();
        assert!(matches!(err, ValidationError::WrongType { .. }));
    }

    #[test]
    fn undeclared_fields_pass_through() {
        // 未在 properties 声明的字段不拦——第一版扁平 schema 下宁可放过
        let args = json!({"path": "a.txt", "extra": true});
        assert!(validate_args("read", &read_schema(), &args).is_ok());
    }
}
