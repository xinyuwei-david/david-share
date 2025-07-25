# 关于GenAI走势的一点思考

**一、从“思考的快与慢”到大模型推理的局限**
在大型语言模型（LLM）的研究与应用中，人们时常借助人类的两种典型思维形态进行类比：
• System 1（快思考）：反应迅速、几乎不经显式推演或分析，主要依靠直觉或经验。
• System 2（慢思考）：执行有意识的逻辑分解与推理，需要投入更多思考资源和时间。

当前大部分知名模型（如 GPT-3.5、Llama2 等）往往只展现出“快思考”的特征——它们在接收输入后直接给出完整回答，速度极快，却不一定具备在内部展开多轮逻辑检验或搜索的能力。如果遇到大型逻辑题、复杂数学运算或需要高准确度的推断场景，缺乏“慢思考”的设计常会导致不可靠的回答。此外，随着互联网上高质量文本数据逐渐被各种团队大规模获取和清洗，预训练阶段也正全面面临“数据匮乏”的难题。在此背景下，如何平衡“思考的快与慢”——即在维持基础对话流畅度的同时，让模型在必要时切换到更深层次的推理模式，成为了新一阶段的研究重点。

**二、训练环节：两条主要 Scaling Law 流派的比较**
语言模型在发展早期主要依托“堆数据”与“堆参数”的方式来增强性能，因此形成了一套称为“Scaling Law”的研究范式，用以探讨在给定算力（Compute, C）条件下，模型参数量（N）与训练数据量（D）之间的投放方式对最终表现（如 Test Loss）可能带来的影响。业内代表性的两条路线来自 OpenAI（Kaplan）与 DeepMind（Chinchilla），因其在参数与数据的配比上理念不同而分化成两大流派。

• OpenAI（Kaplan）流派

- 该路线的核心是：在算力受限的情况下，优先扩充模型参数规模 (N)，数据量 (D) 可保持相对较小比例，即 N : D ≈ 0.73 : 0.27。
- GPT-3 是此思路一个突出的实例：仅使用约 300B token 进行训练，却能在当时刷新各类语言任务纪录，说明“大模型+相对少量数据”也能实现较好的初始收敛。
- 然而，若下一步想继续放大模型而无法匹配更多高质量数据，则可能面临训练不充分的瓶颈，对逻辑推理等要求更高的任务还是力有不逮。

• DeepMind（Chinchilla）流派

- 该路线认为若想最大限度发挥模型参数效能，就必须为每个参数分配尽量充足的 token（常见目标是每个参数分配 20 个 token）。

- 例如针对 70B 参数规模，Ideal 情况应提供 1.4T 以上的高质量 token 进行训练，以避免“饥饿训练”。

- 该方案在理论层面能让模型的潜在能力充分“吃饱”，但实际上超大规模、合规且可用的数据并不容易获得，工程落地难度颇高。

  • 数据瓶颈

- 无论采用哪条路线，想在 2024 年甚至更远的未来再度多倍增长数据规模都相当困难。当前可收集到的互联网文本总量虽表面庞大，但真正高质量、去重、合规且可用的 token 极其有限。

- 包括 DeepSeek-V3、GPT-4.5、Claude、Llama2 等在内的模型均已接近这一“数据墙”。在无法轻易取得大规模新数据的情况下，靠“再度扩大训练集”来提升能力已难以为继。

  **三、“思考的快与慢”在大模型上的映射**
  当“堆数据+堆参数”这条路径接近极限，后续的研究自然转向在推理环节挖掘潜能，即如何让模型不只在一次性生成上下功夫，更能通过多轮搜索、校验与回溯来提升严谨度。用人类思维模式比喻，即从以 System 1（快思考）为主，逐步支持或融合 System 2（慢思考）。

1. System 1（快思考）
   - 现有多数大模型默认执行的一次性输出，可迅速为用户提供回复，对话体验流畅。
   - 然而在数值分析、长链逻辑推理或复杂决策时，缺乏内化推理路径，往往容易出现错误。
2. System 2（慢思考）
   - 引入更严谨的链式推理（Chain-of-Thought, CoT）或搜索过程，类似人类解题时先分解题意、逐步计算及验证，再给出结论。
   - 部分模型通过在训练/微调数据中加入带中间过程的示例已暗含此能力。不过，要想始终稳定且正确地使用 System 2，还需进一步算法或强化学习方式进行加持。
3. 外部 Agent 的局限
   - AutoGPT、MateGPT 等代理系统虽能把一个复杂任务拆分为多轮调用主模型，从而实现“表层多步推理”，但中间步骤的上下文传递和衔接难度较大，也可能产生自我偏移或信息丢失，不如让模型自身展现多步逻辑来得一体化。



**四、The Bitter Lesson 与算力驱动的搜索**

在人工智能发展史上，Rich Sutton 于 2019 年提出的“The Bitter Lesson”指出，大规模使用算力与通用算法的方针往往在实践中大放异彩，而那些过度依赖人工先验知识或精密规则的“小而精”方案随着算力增长会被超越。象棋、围棋以及当代大语言模型的发展历程都印证了这一论断：一旦硬件支持提升，机器倾向于用更强力的搜索或大型模型来获得压倒性优势。


如今，在文本数据难再度翻倍的背景下，一些研究者选择把算力重点投向推理阶段，通过搜索或回溯让模型开展“更深的思考”，从而弥补训练数据不足带来的局限。




**Inference Scaling：多步推理与强化学习**
所谓 Inference Scaling，即在推理时通过多候选并行、回溯评估或强化学习来提高回答质量，远非一次性“快生成”所能企及。

1. 多候选生成与搜索策略
   - Best-of-N：一次性生成 N 条完整回答，再利用自动或人工评分模型选出得分最高者作为最终输出。此法实现相对简易，却无法约束中间步骤，且要占用额外计算资源去生成多个候选。
   - Beam Search：参考机器翻译中的思路，每一步保留若干最优候选继续展开。若在中间步骤引入 PRM（Process Reward Model），则可对每个阶段进行筛选，逐步逼近总分最高的解。
   - Lookahead Search：具备更高级回溯能力，即先向前尝试 k 步，再将评估分值回传给当前节点。此方法能找到更优解，但对算力和时延的需求也更高。
2. 强化学习结构
   - PRM + ORM：前者针对生成过程各个阶段进行评分检查，后者仅对最终结果打分。两者结合可实现多粒度的控制。
   - PPO（Proximal Policy Optimization）：常用于 RLHF（人类偏好对齐），对每个 token 与参考策略的分布差异施加 KL约束，再综合整体回答的质量打分。
   - GRPO（仅 ORM）：省去对中间步骤的建模，仅依赖最终输出的优劣来进行梯度更新，可大大降低显存消耗，但失去对每个生成过程的精细管理。


**五、DS R1：数据有限环境下的多阶段强化实例**



DeepSeek-V3 是一款具备 671B 参数的大型模型，在预训练时已使用约 14.8T token，为突破此类“数据不足”，团队推出 DS_R1，通过多阶段强化学习为模型增加“慢思考”能力。


**初始阶段：R1-Zero**

- 不需要人工标注数据，让模型“自发”输出思考过程，再用自动化规则或过滤策略拒绝明显不合理的推理链。结果虽往往格式欠佳，但保留了大量潜在正确推理片段。


**多阶段强化**

- 冷启动 SFT：先对上述合成推理数据进行监督微调，保证模型输出的可读与规范。

- 推理强化（RL）：继续运用 GRPO 或混合奖励，提升模型多步思考的准确性与连贯性。

- 通用 SFT：融合更多常见对话或问答数据，避免模型陷入“只会长推理，日常对话不流畅”的极端。

- 偏好 RL：在保证逻辑推理的同时，对齐人类价值观与安全标准，过滤潜在有害倾向。

  

评价与不足

- DS_R1 显示，在有限数据环境下仍能通过“自举式”强化学习激发多步推理功能，让模型在复杂任务中给出更精准解答。

- 然而，自动化评分或过滤不一定完美，若奖励设计不当，可能会导入高噪音样本；同时，还需更多安全与合规的 RL 阶段以避免潜在风险。

  
  大多数以“快思考”模式一次性生成答案的模型，常因将“.11”视为“11”而得出“9.11 大于 9.8”的荒谬判断。然而，一旦在 Prompt 中强调“请逐步比较这两个数字的小数位”，就能迫使模型激活链式推理，使其对齐所有小数位，进而正确辨别“9.8 大于 9.11”。这一例子在各类报告中一再被引用，以说明多步推理对防范低级错误的价值。



**六、未来前景：兼顾快思考与慢思考的系统**



由于训练数据的扩增空间有限，很多团队开始关注“如何让大模型在推理时合理安排算力”，既保证日常对话的响应速度，也能在要求高严谨度的场合切换到多步搜索。



• 混合式推理模式：在简单或熟悉的问题上，模型直接给出一次性应答（System 1），维持流畅体验；而遇到推理隐含步骤多、可能出现歧义或高风险的问题时，自动进入多步“慢思考”流程。
• 多模态与强化学习融合：如将视觉、音频、文本等多模态信息综合处理，用更多样化的数据来弥补文本数据的不足，配合更强力的强化学习机制保证搜索过程合理可控。
• 安全合规与价值对齐：随着搜索深度加大，也要防范模型在中间步骤产生不当内容，需要更加精细的 Reward Model 或人工干预策略。

综上所述，为了在训练资源难再扩张的情况下保持大模型的性能提升，研究者们正逐渐将重心从构建更大的预训练数据集，转移到在推理时实现更多步搜索或回溯的“慢思考”机理上。通过引入包括 Best-of-N、Beam Search、Lookahead Search、PPO、GRPO、PRM+ORM 等多种强化学习与搜索方式，模型可在面临复杂问题时展开更完善的逻辑演绎，显著降低出错风险。

DS R1 等工程实例也说明，即使缺乏大规模人工标注或额外高质量语料，模型同样能在后期微调与强化环节中“自举”式地获取多步推理能力。“The Bitter Lesson”中所强调的“通用方法+算力”依旧是大模型发展的主要方向；通过进一步优化在线推理环节的搜索深度与质量，大模型或将在逻辑完整性与应用广度上迎来新一轮提升。