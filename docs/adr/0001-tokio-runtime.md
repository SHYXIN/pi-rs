# ADR-0001: 使用 tokio 作为异步运行时

日期: 2026-10-16
状态: 已接受

## 背景

pi-rs 的核心路径是网络密集型：调用 LLM API 并消费 SSE 流式响应、并发执行工具调用。Rust 需要在项目起点选定异步运行时，这个决定渗透到所有后续代码的形状（函数签名、依赖选型、错误处理），中途更换等于重写。

## 决策

使用 tokio 作为唯一的异步运行时。HTTP 客户端用 reqwest（开启 stream 特性消费 SSE），serde 处理 JSON 序列化。

## 备选方案

- **smol / async-std 系**：更轻量，但生态明显更薄，reqwest 等主流库默认绑 tokio，混用需要兼容层，得不偿失。
- **同步阻塞 + 手动线程**：学习价值高（能看清异步本质），但工程目标下会亲手制造"流式解析 + 并发工具执行 + 取消"三个难题的手工实现负担。

## 后果

- 全库统一 tokio，不引入其他运行时。
- SSE 流式解析基于 reqwest 的 bytes_stream 手写增量解析（OpenAI `data:` 行协议足够简单，不引第三方 eventsource 库，减少一层黑盒）。
- 工具并发执行用 `futures_util::future::join_all` 在同一任务内并发（修订 2026-10-16，hardening：实现实际采用 join_all 而非本条原先写的 `tokio::spawn` + join——工具执行无需跨任务并行，spawn 反而引入生命周期与取消的额外复杂度；REPL 的 Ctrl+C 中断由调用方任务 abort 整体承担）。
