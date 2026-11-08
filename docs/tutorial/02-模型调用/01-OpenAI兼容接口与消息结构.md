# OpenAI 兼容接口与消息结构

本文是第二部分的第一篇正文，它讲清 Agent 如何调用一次大模型。全文分为五节：第一节讲为什么所有节点共用一个调用函数，第二、三节讲客户端与消息结构，第四节讲"无状态"与 Agent 记忆的来源，最后一节给出反例与练习。

> **版本说明。** 本篇的代码片段和行号来自**原版**的 `llm.py`、`graph.py`，用来讲清基本原理。新版的 `llm.py` 改为返回 `LLMResult`（文本、工具调用、token、耗时），并支持 `extra_body` 和 `tools`，见[《改进说明》第 4 条](../../改进说明.md)。原版代码在上一级目录的原仓库里可以对照。

---

## 一、为什么所有节点共用一个调用函数

本项目的 Agent 在一次任务里会多次调用模型：规划一次，每一步选工具各一次，审查一次，总结一次。第一次运行示例任务时，这样的调用一共发生了 **7 次**。

这 7 次调用做的事情各不相同，走的却是同一个函数：`src/cc_agent/llm.py` 中的 `complete`。四类节点的区别，只在于传进去的 system prompt 不同。

| 节点 | 使用的 system prompt | 位置 |
| --- | --- | --- |
| plan | `PLANNER_SYSTEM`：先写一份简短计划 | `src/cc_agent/graph.py:18` |
| act | `ACTOR_SYSTEM`：每次只选一个工具，只输出一个 JSON 对象 | `src/cc_agent/graph.py:23` |
| review | `REVIEWER_SYSTEM`：保守地审查任务是否真的完成 | `src/cc_agent/graph.py:38` |
| summarize | `SUMMARIZER_SYSTEM`：给开发者写运行总结 | `src/cc_agent/graph.py:43` |

> 把模型调用收敛到一个函数里，换模型、加参数、记日志都只需要改一个地方。后面关闭 thinking、记录 token 用量、改造 function calling，都从这里动手。

---

## 二、客户端：只换地址就能换供应商

### 2.1 OpenAI 兼容接口

OpenAI 定义的 Chat Completions 接口已经成为事实标准。硅基流动、DeepSeek 等供应商都实现了同样的请求格式和返回格式，这类接口被称为 **OpenAI 兼容接口（OpenAI-compatible API）**。

因此本项目直接使用 OpenAI 官方的 Python SDK。只要把 `base_url` 指向别家的地址，同一份代码就能调用别家的模型。

`src/cc_agent/llm.py` —— `OpenAICompatibleChat.__init__`

```python
class OpenAICompatibleChat:
    def __init__(self, config: AgentConfig):
        config.require_valid_provider()
        config.require_api_key()
        self.config = config
        kwargs = {"api_key": config.api_key}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = OpenAI(**kwargs)
```

构造函数先做两项检查：供应商必须是 `openai`（即 OpenAI 兼容），API key 不能为空。检查通过后，它把 key 和 `base_url` 交给 `OpenAI(...)`，建出客户端。`base_url` 为空时使用 OpenAI 官方地址；本项目在 `.env` 里把它设成了硅基流动的 `https://api.siliconflow.cn/v1`。

### 2.2 配置从哪里来

`src/cc_agent/config.py` 中的 `AgentConfig.from_env` 先读取项目根目录的 `.env`，再从环境变量里取出 key、地址、模型名、温度和最大步数等配置。

| 环境变量 | 本项目的取值 | 作用 |
| --- | --- | --- |
| `OPENAI_BASE_URL` | `https://api.siliconflow.cn/v1` | 供应商地址 |
| `OPENAI_MODEL` | `deepseek-ai/DeepSeek-V4.1-Flash` | 模型名 |
| `OPENAI_TEMPERATURE` | `0.1` | 采样温度，越低输出越稳定 |
| `CC_AGENT_MAX_STEPS` | `8` | act 最多执行几步 |

> 配置和代码分离：换供应商、换模型都只改 `.env`，不改一行代码。面试问"怎么支持多家模型"，先答这一层。

### 2.3 "兼容"不等于"完全一样"

兼容只保证主干字段一致：`model`、`messages`、`temperature`、`tools` 的写法各家相同。各家自己加的扩展参数并不统一。

以关闭 thinking 模式为例，两家的写法完全不同。

| 供应商 | 关闭 thinking 的参数 |
| --- | --- |
| DeepSeek 官方 | `thinking: {"type": "disabled"}` |
| 硅基流动 | `enable_thinking: false` |

这类参数要通过 OpenAI SDK 的 `extra_body` 传进去，第二篇会详细讲。

> 读者在面试中应当主动说明：OpenAI 兼容降低了换供应商的成本，但没有消除差异，扩展参数仍然要按供应商适配。

---

## 三、一次调用发了什么、收了什么

### 3.1 消息结构

`src/cc_agent/llm.py` —— `complete`

```python
    def complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        content = response.choices[0].message.content
        return content or ""
```

`complete` 每次只发两条消息。第一条的角色是 `system`，写"你是谁、要遵守什么规则"，例如执行节点要求"每次只选一个工具，只输出一个 JSON 对象"。第二条的角色是 `user`，写这一步的具体材料：任务、计划、仓库上下文、工具说明和最近的工具记录。

Chat Completions 接口一共有四种角色。

| 角色 | 放什么 | 本项目是否使用 |
| --- | --- | --- |
| `system` | 角色设定和规则 | 使用 |
| `user` | 用户输入，或者这一步的材料 | 使用 |
| `assistant` | 模型之前的回复 | 未使用 |
| `tool` | 工具执行结果（function calling 时使用） | 未使用 |

> 本项目只用了前两种角色，这正对应"每步重新拼一个单轮 prompt"的设计。第四部分改造 function calling 时，会用上 `assistant` 和 `tool` 两种角色。

### 3.2 返回值里被丢掉的东西

`complete` 只返回 `choices[0].message.content`，也就是回答文本。API 的返回里还有两样重要信息被丢掉了。

- `usage`：这次调用的输入 token、输出 token 和思考 token 数量。
- `reasoning_content`：thinking 模式下模型的思考过程。

连通测试时，一句"用一句话介绍你自己"的返回里，`usage` 显示输出了 **64** 个 token，其中 **45** 个是思考。

> 丢掉 `usage`，就没法统计一次任务花了多少 token、多少钱，也就做不了改进前后的成本对比。第二篇的练习要把它补上。

---

## 四、无状态：Agent 的"记忆"从哪里来

### 4.1 每次调用都是独立的

模型服务端不保存对话。每次调用 `complete`，模型只能看到这一次传进去的两条消息，不记得上一次说过什么。这就是 **LLM 调用是无状态的（stateless）** 的含义。

那么 Agent 是怎么"记得"前面几步做过什么的？答案在调用方：`act` 节点每一步都会把最近的工具记录重新拼进 user 消息，拼接工作由 `_history_text` 完成。

`src/cc_agent/graph.py` —— `_history_text`

```python
def _history_text(history: list[ToolRecord], limit: int = 9000) -> str:
    chunks: list[str] = []
    for item in history[-12:]:
        output = item.get("output", "")
        if len(output) > 1800:
            output = output[:1800] + "\n... output truncated"
        chunks.append(
            json.dumps(
                {
                    "tool": item.get("name"),
                    "arguments": item.get("arguments"),
                    "ok": item.get("ok"),
                    "blocked": item.get("blocked", False),
                    "output": output,
                },
                ensure_ascii=False,
            )
        )
    text = "\n".join(chunks)
    return text[-limit:]
```

`_history_text` 只取最近 **12** 条工具记录，每条的输出超过 **1800** 字符就截断，然后把每条记录序列化成一行 JSON。最后它只保留整段文本的最后 **9000** 个字符。注意 `text[-limit:]` 是按字符截的，可能从某一行 JSON 的中间切开。另外，记录里只有工具名、参数和结果，没有模型当时写的理由（`reason`），所以模型看不到自己之前每一步是怎么想的。

### 4.2 两种记忆方式的取舍

| 方式 | 做法 | 好处 | 代价 |
| --- | --- | --- | --- |
| 单轮重拼（本项目） | 每一步把任务、计划、最近记录拼成一条 user 消息 | 实现简单，长度可控 | 模型看不到自己之前的回复；记录截断后信息丢失 |
| 多轮 messages | 保留完整的 system / user / assistant / tool 消息列表 | 信息完整，模型能看到自己之前每一步的回复 | 列表越来越长，需要压缩 |

> 面试问"LLM 是无状态的，你的 Agent 怎么记住上下文"，按这个顺序答：调用本身无状态 → 记忆由调用方维护 → 本项目用单轮重拼加截断 → 代价是什么 → 改进方向是多轮 messages 加摘要压缩（第五部分）。

---

## 五、反例与练习

**反例。** 假设每个节点各自创建客户端、各自写调用代码。某天要换成硅基流动，开发者得在四个地方改地址和参数；要关闭 thinking，又得在四个地方加 `extra_body`，漏掉一处，那个节点就仍然又慢又贵；要统计 token，四个地方都得补记录，统计口径还可能不一致。

> 这个反例说明，模型调用必须收敛到一个函数：它是换供应商、加参数、记成本的唯一入口。

本篇的练习有两项。

- **练习一：读配置。** 不看代码，说出一次调用用的 `base_url`、模型名和温度分别来自哪里，以及要换回 DeepSeek 官方 API 需要改哪几项。
- **练习二：推演记忆。** 假设把步数上限调高后，某次任务的 act 已经执行了 15 步。第 16 步的 prompt 里还能看到第 1 步的工具结果吗？为什么？

> **指标回指。** 本篇对应第八部分"评测体系"中的平均 token 和平均耗时。目前 `complete` 还没有记录 `usage`，这两项指标暂时没有数据来源；第二篇的练习补上之后才能统计。

---

**本篇涉及的源码：** `src/cc_agent/llm.py`、`src/cc_agent/config.py`、`src/cc_agent/graph.py`（`_history_text`）
