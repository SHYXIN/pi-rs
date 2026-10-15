//! pi-rs CLI 入口。本 ticket 端到端行为：装载配置并打印解析出的连接信息（不含 key）。

use clap::Parser;

use pi_rs::config::{self, DEFAULT_MODEL};

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

    match config::load_connection(&path, &cli.model) {
        Ok(conn) => {
            // 密钥安全：只打印 endpoint 与 model，不打印 key 明文
            println!("config file : {}", path.display());
            println!("model       : {}", conn.model_id);
            println!("endpoint    : {}", conn.endpoint);
        }
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(1);
        }
    }
}
