# Emotion-Aware Movie Recommendation Agent

一个基于 **LLM Tool Calling、结构化用户状态、Memory、BGE 向量检索、FAISS 与可解释重排** 的情绪感知电影推荐系统；系统不会让大模型直接凭印象推荐电影，而是先理解用户当下的情绪、观看目的、类型偏好和避雷项，再从本地电影知识库召回并过滤候选，最后要求模型只根据候选证据生成推荐结果。

## 项目定位

普通电影推荐往往只处理“喜欢什么类型”，但真实请求通常还包含“我今天很累”“想放松一下”“不要太压抑”“节奏别太慢”等即时状态与硬约束，因此本项目将推荐过程拆成理解、召回、过滤、重排和生成五个可观察阶段，使模型负责擅长的自然语言理解与表达，使确定性代码负责检索、约束和排序，从而获得比纯 Prompt 推荐更稳定、可解释且易扩展的结果。

## 整体 Pipeline

```text
用户自然语言请求
        │
        ▼
DeepSeek Agent（决定并组织工具调用）
        │
        ├─ 1. analyze_user_state
        │     抽取 emotion / need / tone / preferred_genres / avoid
        │     更新进程内 Memory
        │
        ├─ 2. rag_retrieve_movies
        │     原始请求 + 结构化状态 → 检索 Query
        │     BGE-small-en-v1.5 → Query Embedding
        │     FAISS IndexFlatIP → 语义候选召回
        │     硬约束过滤 → 排除 heavy / slow / romance 等避雷项
        │     加权 Rerank → 融合语义、类型、情绪、节奏和轻松度
        │
        └─ 3. Grounded Generation
              DeepSeek 基于 candidates 与 evidence 组织推荐理由
        │
        ▼
个性化、可解释的电影推荐
```

一次典型调用的数据变化如下：

```text
"I feel stressed and want a light comedy, but nothing too dark."
    ↓
{
  "emotion": "stressed",
  "need": "relaxing",
  "tone": "healing",
  "preferred_genres": ["comedy"],
  "avoid": ["heavy"]
}
    ↓
语义召回 top_k × 3 个候选 → 硬约束过滤 → 业务特征重排
    ↓
带 retrieval_score、rerank_score、特征明细和 evidence 的候选集
    ↓
基于证据生成最终推荐
```

### 离线索引阶段

1. `agent/build_bge_index.py` 读取 `data/movies.csv`；
2. 将标题、类型、情绪标签、节奏、沉重程度和简介拼成电影文档；
3. 使用 `BAAI/bge-small-en-v1.5` 生成归一化向量；
4. 使用 `faiss.IndexFlatIP` 建立精确内积索引，归一化后等价于按余弦相似度检索；
5. 将向量索引写入 `rag_store/movies_bge.faiss`，将向量对应的业务元数据写入 `rag_store/movies_metadata.pkl`。

### 在线推荐阶段

1. `main.py` 将当前 Memory、用户请求和工具定义发送给 DeepSeek；
2. 模型调用 `analyze_user_state`，将自然语言转为结构化状态；
3. 模型调用 `rag_retrieve_movies`，系统将原始请求与结构化状态共同编码，保留自然语言细节并增强情绪与偏好信号；
4. FAISS 先扩大召回范围，随后以确定性规则执行硬约束过滤，避免语义相近但违反用户明确要求的结果；
5. `tools/ranking.py` 按下式重排候选：

```text
final_score = 0.55 × vector_score
            + 0.20 × genre_match
            + 0.15 × mood_match
            + 0.07 × pace_match
            + 0.03 × lightness
            - constraint_penalty
```

6. 工具将候选电影、检索分数、重排分数、各项特征和原始 evidence 返回模型，模型据此生成最终回答。

## 为什么不使用现成 Agent 框架

本项目没有引入 LangChain、LlamaIndex 等完整 Agent 编排框架，但仍使用 OpenAI-compatible SDK、Sentence Transformers 和 FAISS 等成熟基础组件，因为这里需要解决的是边界清晰、步骤较少的推荐链路，核心流程只有状态分析、检索、过滤、重排与生成，直接使用 Tool Calling 协议和轻量分发代码即可完整表达，引入大型框架反而会增加抽象层、隐式状态和调试路径。

- **流程显式可控**：工具 schema、参数、调用结果和最大循环次数都直接定义在代码中，错误可以准确定位到模型决策、参数解析、检索或排序；
- **业务规则不被框架掩盖**：避雷项过滤和重排权重属于核心推荐逻辑，显式实现更便于解释、调整和做消融实验；
- **依赖与运行成本较低**：当前场景不需要复杂 DAG、分布式执行、通用向量库适配器或多 Agent 协作；
- **协议容易迁移**：编排层依赖通用 Tool Calling 消息格式，后续可以替换模型、检索器、向量索引或 Memory；
- **适合教学与验证**：从 schema、dispatcher 到 tool result 回填模型的全过程均可见，能够直接展示 Agent 的运行机制。

这并不意味着现成框架没有价值，当流程扩展到大量工具、复杂状态机、并行节点、持久化执行、链路追踪或多 Agent 协作时，引入 LangGraph 等编排框架会更合适；当前实现选择的是与问题规模匹配的最小抽象。

## 相比纯 Prompt 工程的优势

Prompt 工程仍然用于约束工具顺序、注入 Memory 和规定基于证据回答，但它不再独自承担数据访问、业务约束和排序逻辑，因此本方案是“Prompt + 可执行系统”，而不是用 Agent 取代 Prompt。

| 维度 | 纯 Prompt 推荐 | 本项目 |
| --- | --- | --- |
| 事实来源 | 依赖模型参数记忆，可能编造影片或属性 | 从本地知识库召回候选并返回 evidence |
| 约束执行 | 自然语言要求可能被忽略 | 代码执行硬过滤，Rerank 再增加惩罚保护 |
| 个性化 | 通常依赖当前上下文 | Memory 累积偏好、避雷项、最近推荐和上次情绪 |
| 可解释性 | 难以判断推荐依据 | 暴露检索分数、重排分数及分项特征 |
| 可测试性 | 输出开放，难做稳定断言 | 每个工具、过滤规则和排序函数均可独立测试 |
| 可维护性 | 规则堆叠在长 Prompt 中，修改容易相互影响 | 理解、检索、过滤、排序和表达分层演进 |
| 数据更新 | 依赖模型已有知识或重新提供上下文 | 更新 CSV 并重建索引即可纳入新电影 |

最重要的差异是，Prompt 只能表达“希望模型遵守什么”，工具链能够真正执行“系统必须保证什么”；例如用户明确避雷沉重影片时，代码过滤属于可验证约束，而不只是对模型的一次提醒。

## 技术选型

| 技术 | 用途 | 选择原因 |
| --- | --- | --- |
| Python | 主开发语言 | NLP、向量检索和数据处理生态成熟，适合快速验证完整链路 |
| DeepSeek Chat | 意图理解、工具选择、推荐生成 | 支持 OpenAI-compatible Tool Calling，便于连接本地工具 |
| OpenAI Python SDK | 模型客户端 | 协议稳定、调用简单，可通过 `base_url` 接入 DeepSeek |
| JSON Schema | 工具参数契约 | 限制必填字段和额外字段，减少参数漂移 |
| BAAI/bge-small-en-v1.5 | 文本 Embedding | 模型轻量，适合当前英文电影数据和本地运行环境 |
| FAISS IndexFlatIP | 向量索引 | 当前数据规模小，精确检索简单且没有近似召回损失 |
| pandas / numpy | 数据与向量处理 | 便于构造元数据并转换为 FAISS 需要的 `float32` 数组 |
| 自定义加权 Rerank | 业务排序 | 显式融合语义相关度、类型、情绪、节奏、轻松度和约束惩罚 |
| 进程内 SimpleMemory | 轻量用户记忆 | 无需数据库即可验证跨调用偏好积累，后续可以替换持久化存储 |

### 为什么选择 BGE-small-en-v1.5

本项目的检索对象不是网页长文，而是由电影标题、类型、mood、pace、heaviness 和简短 description 组成的英文短文本，查询同样是“情绪 + 观看需求 + 类型偏好”构成的短句，因此需要的是擅长语义匹配的专用 Embedding 模型，而不是让生成式大模型直接判断所有电影；生成模型适合理解和表达，但逐条比较候选的成本高、结果不够稳定，也无法像向量索引一样预先计算文档表示并进行高效检索。

选择 `BAAI/bge-small-en-v1.5` 主要基于以下考虑：

- **任务匹配**：BGE 属于面向语义检索训练的文本嵌入模型，可以将“stressed、want to relax”与“warm、light、uplifting”等表面词汇不同但语义接近的内容映射到相近向量空间，比关键词匹配更能覆盖隐含需求；
- **语言匹配**：当前用户示例、电影简介和标签主要为英文，`en` 版本与数据语言一致，没有必要为了尚不存在的多语言需求承担更高模型成本；
- **规模匹配**：`small` 版本体积和推理开销较低，适合课程项目、本地 CPU 环境和小型知识库，同时保留足够的语义表达能力，使用更大的 Embedding 模型在当前十余条数据上收益有限；
- **部署简单**：模型可通过 Sentence Transformers 本地加载，电影文档向量可以离线生成，在线阶段只需编码一条 Query，不依赖远程 Embedding API，因此延迟、费用和数据流向更可控；
- **便于替换和验证**：Embedding 层只负责生成固定维度向量，与业务过滤和重排解耦，后续可以直接替换为 BGE 多语言版、其他开源模型或云端 Embedding 服务，并通过 Recall@K、NDCG 等指标进行横向比较；
- **与 FAISS 配合自然**：索引和查询向量都使用 `normalize_embeddings=True` 归一化，再交给 `faiss.IndexFlatIP` 计算内积，此时内积等价于余弦相似度，既保留语义方向的比较，又不受向量长度影响。

这里没有把 BGE 的相似度直接当成最终推荐分数，因为 Embedding 擅长判断“语义是否相关”，却不能可靠执行“绝对不要沉重内容”等硬约束，也不会天然符合本项目对类型、情绪和节奏的业务权重，所以系统在 BGE 召回之后继续执行规则过滤与可解释 Rerank，形成“BGE 负责召回、规则负责兜底、Rerank 负责排序”的职责划分。

## 目录结构

```text
agent_system/
├── main.py                       # Agent 循环、模型调用、Memory 更新
├── schemas.py                    # Tool Calling 的 JSON Schema
├── dispatcher.py                 # 工具名到 Python 函数的显式分发
├── data/
│   └── movies.csv                # 电影知识库数据源
├── agent/
│   ├── memory.py                 # 进程内用户偏好记忆
│   └── build_bge_index.py        # 离线构建 BGE + FAISS 索引
├── tools/
│   ├── user_state_tool.py        # 用户状态结构化抽取
│   ├── rag_retrieve_tool.py      # 在线向量召回、过滤和结果封装
│   ├── ranking.py                # 业务特征加权重排
│   └── retrieve_tool.py          # 早期纯规则检索基线，保留用于对比
└── rag_store/
    ├── movies_bge.faiss          # 向量索引
    └── movies_metadata.pkl       # 向量对应的电影元数据
```

## 快速开始

### 1. 安装依赖

```powershell
pip install openai pandas numpy sentence-transformers faiss-cpu
```

### 2. 配置 API Key

项目优先读取 `DEEPSEEK_API_KEY`，同时兼容 `OPENAI_API_KEY`：

```powershell
$env:DEEPSEEK_API_KEY="your_api_key"
```

### 3. 构建向量索引

仓库已经包含索引文件，只有在电影数据变化或索引缺失时才需要重新构建：

```powershell
python agent\build_bge_index.py
```

首次运行会下载 `BAAI/bge-small-en-v1.5`，完成后会生成 `rag_store/movies_bge.faiss` 与 `rag_store/movies_metadata.pkl`。

### 4. 启动项目

```powershell
python main.py
```

示例输入：

```text
I feel stressed and want a light comedy, but nothing too dark.
```

终端会打印每次 Tool Call 与 Tool Result，最终输出基于检索证据生成的推荐。

## 核心设计

### Tool Calling 与 Dispatcher

`schemas.py` 是模型可见的工具契约，`dispatcher.py` 是本地执行入口，`tools/` 是业务实现；模型只负责返回工具名和 JSON 参数，真正的数据访问和规则执行始终发生在本地代码中，未知工具、参数错误和执行异常都会被转换为结构化错误并返回上层。

### 双阶段检索与排序

向量召回追求语义覆盖，硬过滤保证明确约束，Rerank 再处理业务偏好，这种“召回宽、约束严、排序细”的设计比单独依赖向量相似度更适合推荐任务，同时每项得分都可以检查和调权。

### Grounded Generation

候选结果携带由本地元数据构造的 `evidence`，系统 Prompt 要求模型依据 evidence 生成理由，因此大模型主要承担自然语言组织工作，不负责决定知识库中不存在的事实，从而降低幻觉风险。

### Memory

当前 Memory 保存喜欢类型、避雷内容、最近候选和最近情绪，并在后续 `run_agent` 调用时注入 system message；它是进程内原型，程序退出后数据会丢失，最近推荐目前也只被记录，尚未进入去重排序逻辑。

## 当前边界与后续演进

- 用户状态分析目前采用英文关键词规则，稳定可解释但覆盖范围有限，可替换为结构化 LLM 抽取或轻量分类模型；
- 电影数据量较小且主要为英文，因此选用 BGE-small-en-v1.5 与 `IndexFlatIP`，扩大数据规模后可切换多语言 Embedding 与 IVF/HNSW；
- Rerank 权重目前为人工设定，可通过标注偏好数据、离线评估和 Learning to Rank 进一步校准；
- Memory 尚未持久化，也未真正执行最近推荐去重，可接入 SQLite/Redis 并将历史反馈纳入排序；
- 当前缺少系统化评估集，后续可增加 Recall@K、NDCG、约束满足率、幻觉率、工具调用成功率与端到端延迟；
- 可增加最终 Verify 节点，检查推荐标题是否来自候选集、理由是否与 evidence 一致、结果是否违反硬约束。

## 一句话总结

本项目的核心不是“让大模型写出更漂亮的推荐 Prompt”，而是将电影推荐构造成一条可执行、可验证、可解释的 Agent Pipeline：LLM 理解需求并组织流程，RAG 提供事实，规则守住约束，Rerank 完成业务排序，最后再由 LLM 基于证据生成自然语言答案。
