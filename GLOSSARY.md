# Glossary

pi-rs 的领域词汇表。只收领域概念，不收实现细节。

---

## Agent Loop（智能体循环）

驱动一次会话的核心循环：用户消息进入 → LLM 生成回复或工具调用 → 工具执行 → 结果回传 LLM → 重复，直到 LLM 给出无工具调用的最终回复。pi-rs 的中枢结构，其他一切（工具、provider、会话状态）都挂在这条循环上。

## Tool（工具）

暴露给 LLM 的一个可调用能力，由三部分组成：`name`（唯一标识）、`description`（给模型看的功能说明，模型据此决定是否调用）、`parameters`（结构化参数 schema）。模型通过 API 原生 function calling 选择工具并生成参数；pi-rs 不做关键词路由。

## Tool Call（工具调用）

LLM 在回复中发出的一次工具使用请求（工具名 + 参数）。由 agent loop 解析、校验、执行，执行结果作为消息回传给 LLM。

## Provider（提供商适配器）

对一个 LLM API 的适配层：把统一的内部消息格式转成该 API 的请求格式（含 tools 声明），把流式响应转回统一的内部事件。第一版只做一个 provider。

## Transcript（会话记录）

一次会话的完整消息历史，包括用户消息、LLM 回复、工具调用与工具结果。是发给 LLM 的上下文的唯一事实来源。

## First Version Boundary（第一版边界）

第一版的完成定义：单 provider + 核心工具集（read/bash/edit/write/grep）+ agent loop + 基础终端 REPL（多行输入 + ANSI 着色），能真实完成代码修改任务。与 pi 的差距（TUI 富渲染、MCP、codemode、durable、多 provider、extension 机制）是后续演进方向，不在第一版边界内。

## Provider Parity Principle（Provider 对齐原则）

以原版 pi 的能力全集为参照系，但第一版只在单 provider（OpenAI 兼容协议）上做穿核心链路；其余能力按「先穿通、再补齐」的顺序逐个 ticket 追平。范围对齐 pi 全集，节奏上允许长期追赶。
