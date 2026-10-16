//! pi-rs CLI 入口（ticket 08：REPL + 真实端到端）。
//!
//! 端到端行为（First Version Boundary）：`cargo run -- --model glm-5.3-flash`
//! 起会话。配置从本机 models.json 装载（密钥零入库，见 config.rs 与 .gitignore）；
//! 工具用第一版五件套注册表；Agent 语义（流式渲染 + 工具循环）见 repl.rs。

use clap::Parser;

use pi_rs::config::{self, DEFAULT_MODEL};
use pi_rs::provider::OpenAiProvider;
use pi_rs::repl;
use pi_rs::tool::{default_registry, ToolContext};

#[derive(Parser)]
#[command(name = "pi-rs", about = "A coding agent in Rust, modeled after pi")]
struct Cli {
    /// 模型 id（对应 models.json 中的条目）
    #[arg(long, default_value = DEFAULT_MODEL)]
    model: String,
}

fn main() {
    let cli = Cli::parse();

    let path = match config::config_path() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    let conn = match config::load_connection(&path, &cli.model) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    };

    // 密钥安全：只打印 endpoint 与 model，不打印 key 明文
    println!("config file : {}", path.display());
    println!("model       : {}", conn.model_id);
    println!("endpoint    : {}", conn.endpoint);
    println!();

    let provider = OpenAiProvider::new(conn);
    let registry = default_registry();
    // 工具工作目录 = 进程启动目录；相对路径以此解析
    let ctx = ToolContext::new(".");

    if let Err(e) = repl::run_session(provider, registry, ctx) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
