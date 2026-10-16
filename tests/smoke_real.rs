//! 真实网关冒烟测试（spec Testing Decisions：显式 `cargo test -- --ignored`
//! 才跑，CI 不依赖）。走真实 glm-5.3-flash 完成读→改→跑测试三步任务。
//!
//! 配置来源：环境变量 PI_RS_TEST_CONFIG 指向含真实 key 的 models.json
//! （与生产同格式）。未设置时测试直接通过并说明跳过原因——保证默认
//! `cargo test` 全绿且零网络。

use std::path::PathBuf;
use std::time::Duration;

use pi_rs::agent::AgentLoop;
use pi_rs::config;
use pi_rs::provider::OpenAiProvider;
use pi_rs::tool::{default_registry, ToolContext};
use pi_rs::transcript::{Message, Transcript};

/// 冒烟测试总超时：真实模型 + 工具循环，给足 5 分钟
const SMOKE_TIMEOUT: Duration = Duration::from_secs(300);

fn test_config_path() -> Option<PathBuf> {
    std::env::var("PI_RS_TEST_CONFIG").ok().map(PathBuf::from)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "真实网关冒烟：需 PI_RS_TEST_CONFIG 指向含真实 key 的 models.json，显式 --ignored 运行"]
async fn real_model_completes_read_edit_test_task() {
    let Some(path) = test_config_path() else {
        eprintln!("skip: PI_RS_TEST_CONFIG 未设置，跳过真实冒烟测试");
        return;
    };
    let conn = config::load_connection(&path, config::DEFAULT_MODEL)
        .expect("PI_RS_TEST_CONFIG 中应有 glm-5.3-flash 条目");

    // 沙箱项目：一个极小的 Rust crate，测试断言故意写错，让模型读→改→验证
    let sandbox = tempfile::tempdir().unwrap();
    let proj = sandbox.path().join("demo");
    std::fs::create_dir_all(proj.join("src")).unwrap();
    std::fs::write(
        proj.join("Cargo.toml"),
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    // add() 的断言是错的（1+1 应为 2），模型的任务是修好并跑测试确认
    std::fs::write(
        proj.join("src").join("lib.rs"),
        "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n\n#[cfg(test)]\nmod tests {\n    use super::*;\n\n    #[test]\n    fn adds() {\n        assert_eq!(add(1, 1), 3);\n    }\n}\n",
    )
    .unwrap();

    let provider = std::sync::Arc::new(OpenAiProvider::new(conn));
    let registry = std::sync::Arc::new(default_registry());
    let ctx = ToolContext::new(&proj);
    let loop_ = AgentLoop::new(provider, registry, ctx);

    let mut transcript = Transcript::new();
    transcript.push(Message::User {
        content: "在这个项目里：cargo test 是失败的。读 src/lib.rs，修复失败的测试（修测试断言，不要改 add 函数），然后跑 cargo test 确认全绿。工作目录就是当前目录。".to_string(),
    });

    let run = tokio::time::timeout(SMOKE_TIMEOUT, loop_.run(&mut transcript))
        .await
        .expect("smoke test timed out after 300s")
        .expect("agent loop failed");

    // 端到端断言 1：沙箱里的断言被改成正确值（模型真的改了文件）
    let lib = std::fs::read_to_string(proj.join("src").join("lib.rs")).unwrap();
    assert!(
        lib.contains("assert_eq!(add(1, 1), 2)"),
        "model did not fix the assertion; lib.rs:\n{lib}"
    );
    assert!(!lib.contains("assert_eq!(add(1, 1), 3)"));

    // 端到端断言 2：transcript 中出现过 cargo test 工具调用且最终成功
    //（模型被要求跑测试验证；宽松断言：至少一次 bash 调用）
    let had_bash = run
        .transcript
        .messages()
        .iter()
        .any(|m| matches!(m, Message::ToolCall { name, .. } if name == "bash"));
    assert!(had_bash, "model never ran bash (cargo test)");

    // 端到端断言 3：最终回复非空（agent 给出了收尾说明）
    assert!(!run.final_reply.trim().is_empty());
}
