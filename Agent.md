LiteAgent 企业级多智能体框架架构规范

1. 核心架构理念

本框架采用严格的三层架构（Skill -> Agent -> Workflow），旨在将大语言模型（LLM）的模糊性限制在局部，用确定性的图计算引擎保障全局业务的稳定性。

隔离原则：智能体（Agent）之间严禁直接通信或共享全量历史对话。

状态双层模型：采用“全局状态机（Global State） + 局部私有上下文（Local Private Memory）”机制。

显式路由：业务流转的分支判断必须由 Workflow 层的确定性代码/图拓扑控制，非 Agent 自主决定。

2. 标准目录结构

未来生成项目时，请严格遵循以下按领域驱动设计（DDD）划分的目录结构：

lite_agent_project/
├── core/                       # 框架核心基础设施（不可轻易修改）
│   ├── base_skill.py           # Skill 基类协议
│   ├── base_agent.py           # Agent 基类协议与私有记忆管理
│   ├── base_workflow.py        # 基于 LangGraph 的图引擎封装
│   └── memory_manager.py       # 上下文隔离与序列化管理
│
├── domain_skills/              # 【第一层】工具技能层（按域划分）
│   ├── search_tools/
│   │   ├── SKILL_google_search.md  # 技能说明契约（喂给LLM的Schema）
│   │   └── google_search_impl.py   # 技能的具体逻辑实现
│   └── risk_tools/
│       ├── SKILL_db_query.md
│       └── db_query_impl.py
│
├── domain_agents/              # 【第二层】智能体层
│   ├── prompts/                # 集中管理的 System Prompts
│   │   ├── risk_agent.txt
│   │   └── support_agent.txt
│   ├── risk_agent.py           # 风控Agent实例（组装 Prompt 与相关 Skills）
│   └── support_agent.py        # 客服Agent实例
│
├── workflows/                  # 【第三层】工作流层
│   ├── states.py               # 存放所有 Pydantic 定义的 Global State Schema
│   └── order_processing_wf.py  # 具体的业务工作流（定义节点、路由边、Mapper/Reducer）
│
├── config.yaml                 # 环境变量与模型配置
└── main.py                     # 系统启动入口


3. 各层级设计契约 (Contracts)

3.1 Skill 层 (技能层)

职责：执行具体的系统级操作（查库、发邮件、调API）。
规范约束：

双文件原则：每一个 Skill 必须由一个具体实现文件和一个 .md 契约文件组成。

MD 契约定义：.md 文件中必须包含：技能名称、一句话描述、输入参数定义（类似 JSON Schema 的自然语言描述）、以及边界限制。系统启动时会解析此 MD 文件动态生成 Function Calling Schema。

无状态：Skill 必须是无状态的纯函数，不保存任何上下文。

3.2 Agent 层 (智能体层)

职责：特定领域的决策大脑，负责意图理解和工具调用。
规范约束：

私有记忆墙：每个 Agent 实例必须在内存中维护自己的 messages 数组。

输入契约：Agent 的输入只能是经过 Workflow Input Mapper 过滤后的局部字典数据，绝不能接收完整的 Global State。

输出契约：Agent 严禁直接修改全局状态。其输出必须是一个标准化的结构体（包含：执行的最终业务结论、消耗的 Token 统计等），交由 Workflow 的 Reducer 去合并。

3.3 Workflow 层 (工作流层)

职责：业务骨架，控制数据流转与状态机推进。
规范约束：

Pydantic 强类型：全局状态（Global State）必须且只能使用 Pydantic BaseModel 定义，所有字段必须有明确的类型和默认值。

Mapper/Reducer 必须纯粹：

Input Mapper：只做数据的剥离和提取（Global -> Local）。

Reducer：只做数据的覆盖或追加（Local -> Global），并在合并时触发 Pydantic 校验。

条件边（Conditional Edges）：路由逻辑必须基于 Global State 中的确定性字段进行 if/else 判断。

4. 给 AI 编码助手 (Claude Code 等) 的系统级开发指令

致 AI 编码助手：
当你读取到本规范准备生成具体代码时，请绝对遵守以下指令：

技术栈限定：使用 Python 3.10+。状态验证强制使用 pydantic v2。底层图引擎优先采用 langgraph。LLM 交互优先使用标准化接口。

禁止 Agent 越权：在编写 domain_agents/ 下的代码时，绝对不要编写任何让 Agent A 直接呼叫 Agent B 的逻辑。一切流转必须在 workflows/ 的图中完成。

严格实现 Mapper/Reducer：在编写图节点（Node）代码时，必须显式编写 input_mapper 和 reducer 函数，严禁将整个 state 对象透传给 Agent。

防御性编程：在 Skill 实现层和 Workflow Reducer 层，必须加入全面的 try-catch。如果 Skill 崩溃，必须返回标准化的错误字符串，而非直接抛出异常导致整图崩溃。

注释规范：所有核心类和接口必须包含 Google 风格的 Docstring，清晰标明其在三层架构中的归属。