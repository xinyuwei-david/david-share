# 深入解析 DeepSeek DeepEP：如何用极致通信优化加速现代 MoE 模型

## DeepSeek几个开源项目

https://github.com/deepseek-ai/open-infra-index/tree/main

### **一、FlashMLA**

FlashMLA 是专为 Hopper GPU 优化的 MLA 解码内核，针对可变长度序列提供优化性能。支持 BF16 精度，并配备块大小为 64 的分页 KV 缓存。据称，它在 H800 GPU（DeepSeek 用于训练 V3 和 R1 的 GPU）上实现了 3000 GB/s 的内存带宽，并在 580 TFLOPS 下达到计算瓶颈。vLLM 已在实现该技术。

接下来，我简单对比一下FlashAttention，paged attention， flashMLA

如果想抓住它们三者“本质”上的区别，可以从以下三个核心问题来理解它们为什么出现、解决什么问题、如何解决——这样就能更直观地看出差异：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. FlashAttention：
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 出现原因：大模型在做Attention时，需要计算一个庞大的注意力矩阵（Q×K维度），不仅计算量大，而且需要存储所有中间结果，显存占用很高。
   • 解决问题：通过“分块（chunk）式”或“流式（streaming）”的算法设计，不用一次性把整个注意力矩阵都存到显存里，而是分块读取、分块计算、分块写回。
   • 关键本质：这是一个“算子级”算法优化——也就是如何在数学计算过程中减少不必要的显存读写。在训练和推理都可以用，可显著降低内存占用并加速计算。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2) Paged Attention / Paged KV Cache：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 出现原因：当我们做推理时，需要存储 Key/Value（K/V）缓存来做自回归（auto-regressive）解码。但很多场景下有多条序列（不同请求）并发，长度还可能不一样，如果用一条大连续缓存，管理和读写都很麻烦，容易浪费显存或频繁拷贝。
• 解决问题：把 K/V 缓存“分页”（类似操作系统对内存分页的概念），让所有序列的 K/V 存取更灵活，谁需要多一点就分给谁，需要少的就占少一点，减少大块连续内存的浪费和调度难度。
• 关键本质：核心在于“缓存管理策略”——它不是去改 attention 本身的数学算法，而是把注意力层需要的 K/V 数据划分成页面，让系统读写更高效。经常跟 FlashAttention 之类的算子搭配使用，二者互不冲突，反而能协同加速。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3) FlashMLA：

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 出现原因：DeepSeek 为了在 Hopper GPU（如 H800）上榨干算力潜力，特别是在推理阶段的Decoder注意力（即“多头注意力解码”）部分做深度优化。FlashAttention 已经够快了，但还是有一定的通用性牺牲了“硬件特定优化”的空间。
• 解决问题：一方面充分利用 Hopper 的特性（例如新的 Tensor Core、BF16 性能、带宽分配等），另一方面针对可变序列长度、分页 KV 缓存做专门的内核代码，以达到更高的带宽利用率和算力效率。
• 关键本质：这是一个“硬件+算子双重定制”方案。它继承了 FlashAttention（减少内存冗余）的思路，又结合 Hopper GPU 的硬件细节进行极限级别的性能调教。结果就是带宽能到 3000 GB/s，算力打到 580 TFLOPS，性能非常夸张。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
总结：
• FlashAttention：本质是“用分块计算来降低注意力时的内存开销和读写量”，在训练和推理都适用。
• Paged Attention：本质是“用分页式 K/V 缓存管理应对多序列或可变长度的请求”，属于缓存管理层面的优化。
• FlashMLA：本质是“贴近特定硬件（Hopper GPU）做的极致Decoder注意力优化内核”，既利用了分块思路也结合分页管理，最终让推理速度和算力利用最大化。

三者各有分工，一个关注算子算法本身（FlashAttention），一个关注缓存管理（Paged Attention），另一个则在硬件上进一步深挖潜力（FlashMLA）。这就是它们从根本上各自解决的痛点和相互配合的方式。



### **二、DeepEP**

DeepSeek R1 的架构采用了 DeepSeekMoE。该通信库专为 MoE 模型的训练和推理设计，优化了全节点和跨节点通信，支持 NVLink 和 RDMA。它包含高吞吐量的训练与推理预填充内核、低延迟的推理解码内核，以及原生的 FP8 分发支持。灵活的 GPU 资源控制使计算和通信能够并行运行。

![Image](https://mmbiz.qpic.cn/mmbiz_png/akGXyic486nW0ib8OHpdskkOZGiaKevxpia5jXaQBTmnbicDgZrQleYwsa5aqhaCFRGszvYy1EbIpvDZn6WFDib5gXqw/640?wx_fmt=png&from=appmsg&randomid=0948ttpt&tp=webp&wxfrom=5&wx_lazy=1)

1. 什么是 DeepSeekMoE / DeepEP？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • DeepSeekMoE：

- “MoE” 是 “Mixture of Experts” 的简称，即“专家混合”模型，它会把特定 Token（输入片段）分配给不同“专家”（Expert）来处理，能让大模型在同样算力下，处理更高维度或更具个性化的特征。
- DeepSeekMoE 指的是 DeepSeek R1 架构采用的 MoE 方案和通信机制整体。

• DeepEP：

- 是一个专门为 MoE（专家并行）设计的通信库，核心目标是如何在多 GPU、多节点之间高效地做全连接通信（All-to-All），把 Token 派发到相应的 Expert，再把结果合并回来。
- 它支持 NVLink （同机多 GPU 间通信）和 RDMA （跨机通信），并且可在训练 & 推理（尤其是推理的 prefilling 阶段和真正的解码阶段）都用上。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2) 为什么这么设计？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• MoE 模型训练和推理最大的“痛点”之一就是大量的全对全（All-to-All）通信：

- 训练时，每个 Token 要找到它对应的 Expert；每个 Expert 处理完结果再合并回来，这个过程牵扯到 GPU 间、节点间大量数据交换。
- 推理时也类似，尤其是在有多 Token、多 Expert 并发的场景，通信负载很大。
  • DeepEP 通过硬件加速 + 通信优化 + 并行调度的方式，尽量减少通信时间，提高吞吐量或降低延迟。
- 同时支持 FP8、BF16 等低精度，加快数据传输和计算速度。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3) 主要表现在哪里？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 高吞吐量（High-throughput）模式：

- 针对训练和推理预填充（prefilling）阶段，DeepEP 提供了“normal kernels”来达到最大的带宽利用。比如集成 NVLink （可达 160 GB/s 级别）和 InfiniBand RDMA（可达 50 GB/s 左右），让多 GPU 或多节点之间数据“狂飙”。
- 这些 Kernel 还能自动处理 FP8 分发（dispatch）和 BF16 合并（combine），对于大批量 Token（比如一次 4096 或更多）能保持非常高的吞吐。

• 低延迟（Low-latency）模式：

- 在推理解码（inference decoding）阶段，DeepEP 用了一套“纯 RDMA+最简内核”的方案，减少额外的 SM 占用，让解码时延能尽可能地缩短。
- 例如 128 或 256 batch size 时，能在百微秒级别的延迟下把 Token 分发到远程 Expert，再把结果合并回来。

• FP8 支持：

- FP8 能让数据量变小、计算加快，但对通信和算子实现要求更高。DeepEP 原生支持 FP8 分发，说明它在极限性能和省显存方面都下了功夫。

• 弹性 GPU 资源控制：

- 在“高吞吐”模式，DeepEP 允许手动指定要用多少个 SM 来专门跑通信算子。
- 这样能做到“边算边传”，不会让通信堵死计算，也不会让计算挤占通信资源。效率更高。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4) 与传统做法或其他通用通信库相比有什么不同？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 不同之处在于“专门为 MoE 量身定制”：

1. 常规的分布式通信库（如 NCCL）在全对全时效率不算太高，因为它更多是针对 All-Reduce、Broadcast 等操作做了深度优化。
2. DeepEP 重点在 All-to-All 上，而且还细分成培训大批量（高吞吐量）和推理小批量（低延迟）两种模式，针对不同场景调优。
3. 内置支持 FP8、BF16、NVLink、RDMA，这些加在一起让这个库在 MoE 环境下更好用。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5) 总结：它能带来什么好处？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 对于大规模 MoE 模型：

- 训练速度更快：大量 Token 进行 Expert 调度时的通信打包、带宽利用都更高。
- 推理时延更低：小批量解码中也能保持相对稳定、迅速的通信过程。
  • 对于运维和部署：
- 能灵活分配硬件资源，比如专用多少 SM 来跑通信、不干扰计算；遇到多节点，还能利用 RDMA 做跨机高速网络，或者配合 InfiniBand 的虚拟通道（VL）做流量隔离。
  • 对于未来扩展：
- DeepEP 还提供了自动调优机制、对自家修改过的 NVSHMEM 有依赖，后续可以针对执行环境、网络情况做更多优化。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一句话来概括：
DeepSeek R1 里面的 DeepSeekMoE / DeepEP 通信库就是“专门为 MoE 模型的训练和推理过程，做大规模 GPU / 多节点之间全对全通信优化的工具”，既能在大批量情况下发挥超高带宽，也能在小批量推理时保持尽量低的延迟，充分利用 NVLink 和 RDMA 等硬件特性，实现性能最大化。



------

### **四、DeepGEMM**

DeepGEMM 是一个支持 FP8 的 GEMM 库，适用于密集型和 MoE 型 GEMM 操作，支持 V3/R1 的训练和推理。它在 Hopper GPU 上实现了超过 1350+ FP8 TFLOPS 的性能，同时保持轻量化，没有繁重依赖。该库完全采用即时编译（Just-In-Time），核心逻辑仅约 300 行代码，却在大多数矩阵规模上超越了专家调优的内核。它支持密集布局和两种 MoE 布局，为不同工作负载提供灵活性。

![Image](https://mmbiz.qpic.cn/mmbiz_png/akGXyic486nW0ib8OHpdskkOZGiaKevxpia5ibj89Yy29A3lTQUeNkj78EmEBT64JiaDvz1QgekFsNp3KpD2th7XpU0Q/640?wx_fmt=png&from=appmsg&randomid=7emqvrji&tp=webp&wxfrom=5&wx_lazy=1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 什么是 DeepGEMM？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • DeepGEMM 是一个专门针对 FP8 精度做矩阵乘法（GEMM）运算的轻量级库。
   • 和传统的通用 GEMM 库相比（如 cuBLAS 或者基于 CUTLASS 实现的内核），它在 Hopper GPU 上能跑到 1350+ FP8 TFLOPS，性能非常高。
   • 它的源码结构很简单，关键内核逻辑仅 300 行左右，而且采用了“即时编译（Just-In-Time）”的方式来生成最优代码，避免繁重的编译或依赖。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2) 为什么需要 DeepGEMM？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 近来流行的 FP8 精度可以进一步减小张量的存储量并加快计算过程，对大模型尤其是 MoE（Mixture of Experts）场景非常有用。但常规 GEMM 内核对 FP8 的支持并不完善，或是需要复杂的修改。
• 在 DeepSeek V3/R1 中不仅涉及大规模密集（dense）矩阵，还涉及到专家并行（MoE）时的分组矩阵乘法，很多库不一定对这些形态做了深度优化。
• DeepGEMM 就针对那些常见于 MoE、推理或训练过程的矩阵规模（比如 M、N、K 维度以及分组数）做了高度定制，直接用最简洁的方式获得高性能。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3) 它有哪些亮点功能？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 支持密集型和 MoE 型 GEMM：可处理传统的全局 dense 矩阵，也能处理分组式、带掩码的 MoE 场景（比如不同专家分配不同 Token）。
2. 全部基于 JIT 编译：
   • 安装时不需要把所有可能的形状都提前编译；
   • 运行时会根据实际的 M、N、K、大块尺寸等参数自动生成内核并编译，完全展开 MMA（Matrix Multiply-Accumulate）管线，性能更好。
3. 轻量化：
   • 代码量小，逻辑简单，维护起来方便；
   • 对 CUTLASS、CuTe 一些概念做了借鉴，但不会过度依赖其模板或大型抽象。
4. 针对 Hopper 设计：
   • 充分利用了 Hopper 的 TMA（Tensor Memory Accelerator），通过 TMA load/store、multicast、prefetch 等技巧让数据传输效率最大化；
   • 采用 CUDA Core 做“双层累加”（promotion），弥补 FP8 在张量核（Tensor Core）内部累加精度不足的问题。
5. 灵活性：
   • 同时支持不同 MoE 布局（“连续布局”和“掩码布局”），可以应对训练阶段（大批量 Token）和推理解码阶段（小批量变长 Token）；
   • 无需过度手工调参，库里自带一套自动选择最优配置的策略。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4) 实际性能如何？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 在官方测试里，对 DeepSeek 场景常见的各种矩阵大小（比如 4096×7168、128×2112×7168 等）都做了验证。
• 大多数情况下，DeepGEMM 的速度要比“基于 CUTLASS 的内部专家版本”更快，可能快到 1.1x ~ 2.7x 不等，有些形状甚至可以翻倍提升。
• 在典型的 MoE 分组场景下，也能维持 1000+ TFLOPS 的 FP8 计算速度。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5) 使用场景

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 训练或推理时的密集型 GNN/Transformer/BERT 等，需要高效的 FP8 矩阵乘法；
• MoE 模型中，尤其是大批量训练、prefilling（推理前置计算）和小批量的解码阶段都可以调用 DeepGEMM 做分组或掩码式 GEMM；
• 配合 DeepEP 之类的通信库一起用，完成上层分发（dispatch）和合并（combine）后，再用 DeepGEMM 做专家内部的 FP8 乘法，加速端到端流程。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

6) 总结：为什么值得关注？

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• DeepGEMM 面向 FP8 和 MoE，解决了“高性能矩阵计算”和“代码简单灵活”两难的问题。
• 通过 JIT 编译 + Hopper TMA 优化，它能在短小代码里跑出超专家调优级别的性能表现。
• 对开发者来说，既可以直接调用其 Python 接口，也可将其融入更大规模的分布式训练或推理流水线，轻松享受 FP8 的优点。

一句话概括：DeepGEMM 就是一个极简但能爆发高性能的 FP8 GEMM 库，既可以搞定常规场景，又能灵活适配 MoE 的分组需求，让 Hopper GPU 的算力被充分释放。



### **五、DualPipe 和 EPLB**

在专家并行（EP）中，不同专家被分配到不同 GPU 上，可能会导致工作负载不平衡。DeepSeek-V3 通过“冗余专家”策略解决了这一问题，复制高负载专家并以启发式方法分配以达到平衡。组限制专家路由（Group-Limited Expert Routing）通过将同组专家尽可能放置在同一节点上，进一步减少跨节点数据流量。开源的 `eplb.py` 算法计算了平衡的专家复制与分配。他们还引入了 DualPipe，这是一种双向流水线并行算法，能够完全重叠正向与反向计算-通信阶段，同时减少流水线气泡。

![Image](https://mmbiz.qpic.cn/mmbiz_png/akGXyic486nW0ib8OHpdskkOZGiaKevxpia5JkymXBKoxdlmicB25xvAG0qTp9pyJgVB386DW91y6et371K0KMvjibLQ/640?wx_fmt=png&from=appmsg&randomid=34qxmuiq&tp=webp&wxfrom=5&wx_lazy=1)


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 背景：Expert Parallelism 为什么需要“负载均衡”？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 在 MoE（专家混合）模型中采用 Expert Parallelism（EP）时，每个“专家”被分配到不同的 GPU 上处理不同的 Token。
   • 问题在于：专家处理的工作量可能不一样（有的专家很忙，有的专家空闲），会导致 GPU 之间负载不平衡，整体训练或推理吞吐就被拉低。
   • 深度学习系统规模一旦变大，这种不均衡就会更明显，影响模型效率。

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. EPLB：如何解决专家负载不均衡？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 重复专家（redundant experts）策略

- DeepSeek-V3 在遇到“过热”的专家时，直接把它复制多份，让更多 GPU 协同承担这个专家的工作，从而分摊压力。
  • 组限制的专家路由（group-limited gating）

- 把同一专家组（group）的专家偏向放在同一个节点里，减少跨节点间的数据传输；

- 能减轻网络带宽压力，训练/推理更顺畅。
  • EPLB 开源算法（eplb.py）

- 它会根据每个专家的“预估负载”来计算如何复制、如何在节点和 GPU 上分配；

- 核心目标就是让各个 GPU 的负载尽量“平衡”，既提高硬件利用率，也降低网络通信开销。
  • 分层（hierarchical）和全局（global）两种策略

- 分层策略：适用于“专家组的数量能够被节点数整除”的场景，会先做组到节点的平衡，再做节点内复制和分配；

- 全局策略：适用于更松散情况，先无视组的概念，全局地复制和放置专家，然后再分配到 GPU 上。

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. DualPipe：为什么要“正反向双向流水（bidirectional pipeline）”？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 传统流水并行（pipeline parallelism）

- 通常把网络的不同阶段拆开，先做完前向（forward）再做后向（backward），或者在一定程度上分多个微批次（micro-batch）流水化。

- 缺点：在前向和后向转换时，可能出现“流水空泡”（pipeline bubble），即资源闲置没被充分利用。
  • DualPipe 的思路

- 将前向和后向计算-通信阶段做最大程度的重叠（overlap），让 GPU 在做前向计算的同时，可以开始后向通信，或者另一端正在做后向计算时，前向也在通信等等。

- 这样“正向流水”和“反向流水”可以在时间轴上交错执行，减少等待时间，降低流水空泡。
  • 好处

- 提升整体吞吐：更少的泡泡（idle time）意味着更多算力被利用；

- 内存压力合理：相比传统流水或 1F1B 策略，DualPipe 的阶段划分和调度能控制激活（activation）及参数的峰值占用。

- 对大规模分布式训练尤其有用，平时网络带宽与计算都很“吃紧”，DualPipe 这一招能让网络传输和计算叠加起来，减少总时长。

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 实际上怎么用？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • EPLB 部署：

- 先根据历史运行统计或人工估计，得到每个专家的负载；
- 把负载信息传给 eplb.py（或者你自己封装的接口），它就会输出专家复制（replication）和放置（placement）计划；
- 在 DeepSeek-V3 的框架里，这个结果会被用来初始化模型的 Expert 并行部分。
  • DualPipe 部署：
- 在进行流水并行（pipeline parallelism）的时候，用 DualPipe 算法的调度策略，安排前向和后向的计算/通信顺序；
- 需要在代码里实现 overlapped_forward_backward 这样的接口（例如在官方示例 example.py 里可见），把 DualPipe 的调度嵌进去。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

5) 总体收益

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• 整体训练或推理效率提升：

- EPLB 通过负载均衡，让每块 GPU “各司其职”，不再出现“某块卡特别忙，其他卡闲着”的浪费；

- DualPipe 把前向后向流水化到极致，减少等待、压缩空泡。
  • 资源利用率更高：

- GPU 算力、网络带宽都会被恰到好处地分配和调度。
  • 规模可扩展：

- 面向多节点时，Group-limited routing + DualPipe + EPLB 的组合也能发挥出大规模水平扩展的威力。

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  一句话概括：
  EPLB + DualPipe 就是 DeepSeek-V3 在 Expert Parallelism 和流水并行层面上的“双管齐下”优化方案：一个专注于解决“专家负载不均衡”，另一个专注于“把前向后向流水尽可能重叠”，两者共同提升模型训练/推理的整体效率。



------

### **六、Fire-Flyer 文件系统（3FS）**

Fire-Flyer 文件系统是为 AI 工作负载设计的高性能分布式文件系统。它利用 SSD 和 RDMA 网络实现共享存储，简化分布式应用开发。它支持数据准备、随机访问数据加载器、高吞吐量检查点，以及推理的 KV 缓存。

![Image](https://mmbiz.qpic.cn/mmbiz_png/akGXyic486nW0ib8OHpdskkOZGiaKevxpia5iak4JJykeNL9fEBE6XWadEaV92BHPawwmqQBTgFzc7LFOQRImE4Vic7w/640?wx_fmt=png&from=appmsg&randomid=0g3ykoza&tp=webp&wxfrom=5&wx_lazy=1)

1. 什么是 Fire-Flyer File System (3FS)？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 3FS 是一个面向 AI 工作负载的高性能分布式文件系统。
   • 它以 SSD + RDMA 为核心，提供一层共享存储，让所有计算节点都可以方便地读取、写入海量数据；
   • 与常见的分布式存储相比，3FS 专门针对大规模训练、推理等场景做了优化，比如多节点并发读写、大批量随机访问、海量 checkpoint 等。

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. 它有什么亮点特性？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Disaggregated 架构

   - 把许多 SSD 组合成一个大容量、高吞吐的存储池，再配合高速网卡（如 InfiniBand），使得计算节点不必局限于本地存储，就能获取远端 SSD 的高带宽。
   - 应用层不用过度关心“数据到底在哪个节点上”，减少了存储热点和本地/远端访问差异。

2. 强一致性 (CRAQ)

   - 通过“链式复制 + 分配查询 (Chain Replication with Apportioned Queries)”来实现强一致性，保证数据不会出现“读到脏数据”之类的问题；
   - 对分布式应用来说，这能简化很多并发读写的逻辑。

3. 标准文件接口

   - 用户可以像操作本地文件夹一样管理大规模数据，不必学习新的 API；
   - 底层利用了事务型 KV 存储（比如 FoundationDB）来追踪元数据，保证操作的一致性和可扩展性。

4. 针对 AI 的多种功能场景

   - 数据准备（Data Preparation）：方便组织预处理后的数据、管理中间输出；

   - 随机访问 Dataloader：支持随机抽取训练样本，减少重复搬运、支持分布式并发；

   - 高吞吐 Checkpoint：可以在大规模训练时快速保存模型参数；

   - KVCache（Key-Value 缓存）用于推理：
     • 推理时常见的 KV 缓存（例如存储 decoder 的 K、V 矩阵）量极大，如果用 DRAM 容量会很昂贵；
     • 3FS 提供“SSD + RDMA”式 KVCache，能在大量数据场景下保持不错的带宽和读写延迟，同时成本更低。

     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 为什么需要 3FS？
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 大模型训练/推理会产生极其庞大的数据读写需求（训练集、预处理中间结果、模型参数/梯度以及推理时需要的 KVCache）。
   • 普通存储系统如果没有高并发能力，或者没有为 AI 训练的工作模式做适配，可能会成为瓶颈。
   • 3FS 利用了现代 SSD、RDMA 等硬件组合，再加上一些分布式协议上的改进，可以让多节点同时高效读写海量数据，把存储瓶颈降到最低。

   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2. 性能和实际效果
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 3FS 官方给出了多节点大规模测试数据：

- 180 多个存储节点，数百块高带宽网卡，最终汇总可达到 6.6 TiB/s 的吞吐量；

- 在 GraySort 等基准测试中，也展现了非常可观的数据处理速度；

- KVCache 读速度峰值可到 40 GiB/s，同时可以配合后台 GC（垃圾回收）处理删除操作。
  • 这些指标说明：对于大规模训练或推理场景，3FS 可以充分发挥网络带宽和 SSD 的潜力，并且具备良好的一致性和易用性。

  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 社区和影响
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   • 据说 DeepSeek AI 的大模型 R1 在训练时仅花了数百万美元的计算成本，有赖于他们一系列“低开销、高效率”的软件栈，3FS 可能就在其中扮演了重要角色。
   • 3FS 也帮助他们在推理服务上实现非常低价的 API 接口；同时，这些库和内核开源后，在社区上已经获得了上千甚至上万的 star，可见其影响力。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
一句话总结：
Fire-Flyer File System (3FS) 就是一款针对 AI 训练和推理打造的高性能分布式文件系统，用 SSD + RDMA 提供一层高带宽、强一致的共享存储，并内置了专门支持数据准备、随机 dataloader、高速 checkpoint、KVCache 等多种 AI 核心需求。对大规模分布式训练或推理来说，3FS 能显著简化开发、提升吞吐，同时保持相对低的使用成本。

## DeepEP详解

**DeepEP = 精确分桶 + 多路 P2P 并发 + 后台 DMA**
在 H100-NVLink3 + IB-NDR400 环境下，只需一行

```
recv, hdl, _ = deep_ep.Buffer.dispatch(...)
```

即可把原本阻塞的 NCCL All-to-All 换成异步 P2P 传输，并与计算深度重叠：
– 训练主路径几乎不再受 NIC 吞吐限制
– 自回归推理（batch = 1，≤2 k ctx）单 token 延迟可低至 **0.4 ms**



也就是说：DeepEP 把一次同步 **NCCL All-to-All** 通信拆成 **同一 kernel 内并发触发的多条 GPU-to-GPU DMA**：
Router 先把 *必须跨卡* 的 token 按目的 GPU **分桶**并 **FP8 压缩**成小包，然后通过 NVLink 或 RDMA 后台直写。

- 传输数据量 60–70 %

- 纯通信往返开销 **< 0.15 ms**

- 单 token 完整推理延迟降至 **0.38–0.45 ms**
  通信几乎不再是 MoE 瓶颈。

  

1. 什么是 bucket？

在 Dispatch 之前，Router 会把「要发往同一目标 GPU 的 token」先收集到一个临时的小袋子里。
这个装同一目的地数据的小袋子，就俗称 **bucket（桶）**。
换句话说：
• bucket→G1 = “准备发给 GPU-1 的那一批 token”
• bucket→G2 = “准备发给 GPU-2 的那一批 token”
……

举例场景中一共有 4 块 GPU（GPU-0 … GPU-3）。
Router-0 把自己手里的所有 token 逐个查看后，发现它们需要的专家分布在 4 块 GPU 上，于是就分出了 4 个桶：

```
bucket→G0  bucket→G1  bucket→G2  bucket→G3
```

这样做的好处：一旦进入通信层（all-to-all / nvshmem_put），发送方只需把每个桶整体丢给对应的 GPU，对端就能一次性收下属于自己的部分，减少拆包组包开销。



下面是一段**纯举例**的路由结果（方便说明概念，真实数字每个 batch 都会变）：

| token ID | Router 选中的第一专家编号 | 该专家所在 GPU |
| -------- | ------------------------- | -------------- |
| 0        | #12                       | GPU-0 (本地)   |
| 1        | #78                       | GPU-1          |
| 2        | #130                      | GPU-1          |
| …        | …                         | …              |

统计下来也许得到：

- 有 **60 个 token** 选中的专家都在本地 GPU-0
  → 放进 `bucket→G0`，其实留在本地不用真发，图里就写 “(本地60tok)”。
- 有 **20 个 token** 需要去 GPU-1
  → 放进 `bucket→G1 (20tok)`。
- 有 **15 个 token** 需要去 GPU-2
  → 放进 `bucket→G2 (15tok)`。
- 有 **5 个 token** 需要去 GPU-3
  → 放进 `bucket→G3 (5tok)`。

不同 bucket 里的 token 数完全取决于路由打分的结果，所以它们大小不一样很正常。

**小结**

“bucket→G0 (本地60tok)” 这一行只是在示例里告诉你：
Router-0 决定有 60 个 token 要交给 **本机** 的专家，所以放进属于自己 GPU 的 bucket；其余 40 个 token（20+15+5）要发到别的 GPU，于是分成三个小桶分别寄出。

实际运行时，每个 batch、每个 GPU 的 bucket 数量都会动态变化，但流程完全一样。



技术术语：

| 缩写               | 全称                       | 本文语义                              |
| ------------------ | -------------------------- | ------------------------------------- |
| MoE                | Mixture-of-Experts         | 稀疏激活的大模型结构                  |
| Expert             | —                          | 通常是一段两层 MLP (FFN)              |
| Dispatch / Combine | —                          | token→Expert / Expert→token 的通信    |
| FP8                | 8-bit float                | E4M3 或 E5M2，DeepEP 只在链路上传输用 |
| NVLink             | NVIDIA GPU 互连            | ≈ 150 GB/s（Hopper 单向理论）         |
| RDMA               | Remote DMA over InfiniBand | ≈ 35–38 GB/s（400 GbE CX-7 实测）     |
| SM                 | Streaming Multiprocessor   | GPU 里的“核心簇”                      |
| QP                 | Queue Pair                 | RDMA 中的“通信通道”                   |



**DeepSeek的数据类型**

“正向用 FP8、反向梯度用 FP16”是当下几家大模型框架（DeepSpeed、Megatron-LM、NVIDIA Transformer Engine 等）常见的一种 **混合精度训练策略**，核心目的是在**不牺牲精度的前提下进一步压缩显存与带宽**。下面把几件容易混淆的事拆开说明：

1. FP8 是算什么的？
   • 目前工业界主流的 FP8 有两种格式：`E4M3` 和 `E5M2`（分别 4 位或 5 位指数，3/2 位尾数）。
   • 典型使用：
   – **正向权重、激活张量** → FP8 存储 + FP8 TensorCore 计算
   – **梯度张量** → 仍保留 FP16/BF16
   – **优化器状态** → 通常 FP32（Adam m/v）或 8-bit 量化（Adafactor 也有 16-bit 方案）
2. 为什么反向不用 FP8？
   • 梯度分布更尖锐、更容易在极低位宽下溢出 / 欠精度。
   • 业界实测在 GPT-3 / GPT-NeoX 规模上：
   – 正向 FP8 + 反向 FP16 → 收敛曲线与全 FP16/BF16 几乎重合
   – 正反都 FP8 → 需要复杂的 loss-scaling、分段 re-casting 才能收敛，收益不大
3. 硬件要求
   • **A100（Ampere）**：TensorCore 支持 FP16/BF16/TF32/INT8，但不支持原生 FP8；想用 FP8 只能在 CUDA 里做软件 emulation，几乎没加速效果。
   • **H100（Hopper）/GH200**：原生 FP8 TensorCore，峰值 4 PFLOPS（FP8, 2:1）；所以大多数“FP8 正向”案例默认你手里是 H100。
   • **其他加速器**：例如 AMD MI300A 支持 FP8 (E4M3)，但生态仍在补。所以如果你用的是 **“FP16 加速器”**（如 A100），就算代码里写了 FP8，最终也会 fallback 到 FP16 路径，既得不到速度，也得不到显存节省。
4. DeepSeek / DeepSpeed / Transformer Engine 的差异
   • 如果你指的是 **DeepSpeed**（微软开源框架），它的 **FP8 training** 模块就是“正向 FP8 / 反向 FP16”。
   • **DeepSeek**（非微软、做检索/模型开源的那个团队）目前公开 repo 里并没有 FP8 训练代码，多数还是 FP16/BF16 混合精度。
   • **NVIDIA Transformer Engine** 则提供了 E4M3/E5M2 的 kernel，实现方式与 DeepSpeed FP8 基本一致。
5. 总结对应关系
   | 流程阶段 | 张量类型 | 建议位宽 | 硬件依赖 |
   | -------- | -------- | -------- | -------- |
   | 正向 | 权重 / 激活 | **FP8** (E4M3/E5M2) | H100/Hopper 必须 |
   | 反向 | 梯度 | **FP16** 或 BF16 | A100 及以上都 OK |
   | 优化器 | m/v, 统计 | FP32 或 8-bit 量化 | 与显卡无关 |

若你正在用 A100 或其他仅支持 FP16 的 GPU，想尝试 FP8，基本只能获得“学术实验”价值；真正的显存/带宽/速度收益须等到手上有 Hopper 级硬件。

## DeepSeek V3/R1推理示意图

我们先看V3/R1 整体架构如下：

``` 
Input Token IDs
      │
      ▼
┌────────────────────────────────────────────────────────┐
│                    Embedding Layer                     │
│  • Token Lookup (vocab → 7168)                         │
│  • Rotary Positional Encoding (RoPE)                   │
└────────────────────────────────────────────────────────┘
      │   (hidden size d = 7168)
      ▼
══════════════════════  Dense Block 1  ══════════════════════
│   Pre-LayerNorm Transformer – “warm-up” layer ①            │
│   Purpose: stabilize features before sparse MoE part       │
│  1. LN-1                                                   │
│  2. Multi-Head Attention   (128 heads)                     │
│  3. Residual Add                                            │
│  4. LN-2                                                   │
│  5. Feed-Forward  Linear → GELU → Linear (dense FFN)       │
│  6. Residual Add                                           │
═════════════════════════════════════════════════════════════
      │
══════════════════════  Dense Block 2  ══════════════════════
│   Same structure as Block 1 – “warm-up” layer ②            │
═════════════════════════════════════════════════════════════
      │
══════════════════════  Dense Block 3  ══════════════════════
│   Same structure as Block 1 – “warm-up” layer ③            │
═════════════════════════════════════════════════════════════
      │
══════════════════════  MoE Block 4  ════════════════════════
│  1. LN-1                                                  │
│  2. Multi-Head Attention  (128 heads)                     │
│  3. Residual Add                                          │
│  4. LN-2                                                  │
│  5. Router / Gate  (Top-8 selection)                      │
│     ├─ Dispatch  →  send token buckets across GPUs        │
│     ├─ Expert Compute:                                    │
│     │    • 1  Shared Expert                               │
│     │    • 256 Routed Experts  (Linear-GELU-Linear)       │
│     └─ Combine  ←  gather outputs back to source GPU      │
│  6. Residual Add                                          │
═════════════════════════════════════════════════════════════
      │
      ▼
……  MoE Block 5  …  MoE Block 61   (58 identical MoE blocks) ……
      │
      ▼
┌────────────────────────────────────────────────────────┐
│                   Final LayerNorm (global)             │
└────────────────────────────────────────────────────────┘
      │
┌────────────────────────────────────────────────────────┐
│     LM Head   (7168 → 129 280 logits, weight-tied)     │
└────────────────────────────────────────────────────────┘
      │
Softmax / Sampling  →  next-token prediction
```

**Key notes:**

1. Dense Block 1-3 为密集的 "warm-up layers"：完全的 dense transformer blocks，通过完整的前馈网络达到在 MoE 路由前稳定特征表达的目的。
2. 从 MoE Block 4 开始直至第 61 层，每层都用 Router 和 MoE 专家（1 个共享专家+256 个路由专家）代替 dense FFN，同时引入跨 GPU 通信（两次跨卡通讯： Dispatch / Combine）。
3. 最终的全局 LayerNorm 和 LM Head 将隐藏状态转为最终的词表 logits。LM Head 通常权重与初始 Embedding 层共享。

### 推理的两个阶段与具体特点

模型推理分为两个阶段：**Prefill 阶段** 和 **Decode 阶段**:

![images](https://github.com/xinyuwei-david/david-share/blob/master/Deep-Learning/DeepEP/images/6.png)

#### （1）Prefill 阶段：

- 一次性并行处理所有输入 Token，生成每个 Token 的隐藏状态，用于后续生成阶段或其它任务。
- 不需要读取 KV-Cache，但会为 Decode 阶段创建和存储 KV-Cache。

![images](https://github.com/xinyuwei-david/david-share/blob/master/Deep-Learning/DeepEP/images/4.png)

#### （2）Decode 阶段：

- 自回归逐个生成输出 Token，每次仅输入一个新 Token，使用 Prefill 阶段的 KV-Cache 进行加速。
- 每一层模型的 Self-Attention 会利用已有的 KV 缓存加速计算过程，减少重复计算开销。

![images](https://github.com/xinyuwei-david/david-share/blob/master/Deep-Learning/DeepEP/images/5.png)

#### Prefill 与 Decode 阶段均会完整通过 61 层

无论 Prefill 阶段还是 Decode 阶段，都必须完整地依次流过**全部61层结构**：

- **Prefill 阶段:**
  一次性并行处理全部输入的 Prompt 序列，所有 Token 批量通过全部 61 层 (3 层 Dense + 58 层 MoE)，为每个 Token 生成隐藏状态，并创建和存储KV-Cache以用于 Decode 阶段。

  Prefill 阶段特点：

  - 可高效利用 GPU 并行加速。
  - 对于每个 Token，**同样地只激活 8 个路由专家 + 1 个共享专家**（同样走稀疏MoE)，区别仅在于 Prefill 阶段通常 Prompt 序列较长，Token 数量较多，每批处理的 Token 数量可能更多。
  - 在这个阶段，由于并行处理的序列较长，为提高负载均衡效率和吞吐，采用了 EP32 / DP32 配置（较少节点、较多专家冗余部署，每GPU 9 路由专家）。

- **Decode 阶段:**
  Decode 是逐个 Token 自回归生成的过程，每次仅产生 1 个 Token。但每生成一个新的 Token，都需完整经过所有 61 层计算(包含MoE层)，而且每次计算需要读取 Prefill 阶段存储的 KV-Cache，顺序按层执行。

  Decode 阶段特点：

  - 每次只处理少量 (通常仅1个）新 Token，实时逐步生成。 -**仍然只激活 8 个路由专家 + 1 个共享专家**（同样走稀疏MoE)。但每次只处理一个 Token 的情况与 Prefill 阶段的一大批 Token 并发不同，Token 数少，吞吐要求会更高，因此对专家进行了更分散的部署和更多节点的调度，因此使用了 EP144 / DP144 配置（较多节点，每 GPU 仅2个路由专家）。

#### Prefill 与 Decode 环节都要使用 MoE 架构

实际上，从计算架构层面来说，Prefill 与 Decode 两个阶段本质并无区别：

- 都依次经过了完整的61层。
- 每一个 MoE Block （第4至61层）内，都是通过门控路由选择 8 个路由专家+1 个共享专家去进行稀疏激活计算，无论你是一次处理很多 Token (Prefill)，还是每次只处理一个 Token (Decode)，流程都是如此。
- 区别仅仅在于 Prefill 阶段一次性输入大量 Token，更注重高效“并行批处理”；Decode 一次少量 Token，更关注低延迟快速响应（逐步生成新词）。

因此架构设计本身：
**不用对 Prefill 和 Decode 阶段分别设计不同的 Transformer 层结构**。只需在不同阶段调整一下部署配置规模（专家冗余程度和节点数）即可实现高效推理。

------

#### “每层都有 Attention Head” 和 “LM Head 只在最后” 在两个阶段同样适用：

- 每层中的 **Attention Head (128 head)**，无论 Prefill 还是 Decode，都是在每个层的 Attention 子模块中需要计算的。
- **LM Head** 最终只出现在模型最后。此结论同样适用于 Prefill 和 Decode 两阶段。Prefill 后给出隐藏状态，无需LM Head；Decode阶段，每次完成一次全层计算后，经最后一层的 LM Head 投影 获得 logits 进行下一个预测。"**LM Head**" **不属于** Transformer 结构内的这**61层**，而恰好位于全部Transformer层都结束后的“额外投影层”；因此Prefill阶段不需要LM Head并不意味着少算了一层，仅表示“最后的额外投影不需要在Prefill做”。

------

## 总结

回答你最初的问题：

> **“MoE Block 4到61层在 Prefill 与 Decode 阶段是如何体现的？”**

核心结论就是：

- 在 **Prefill 阶段**，模型并行批量处理 Token，MoE Block 每层内对每个 Token 激活 8 个路由专家 + 1 个共享专家，然后生成缓存隐藏状态，批处理加速。
- 在 **Decode 阶段**，每次生成1个 Token，同样经过全部 61层 MoE 层，并从此前缓存的KV Cache中加速计算，同样每层激活8个路由与1个共享专家，但更强调实时生成，采用更多节点、更分散的专家部署来提高吞吐和减少执行延迟。

因此，模型的结构并没有在Prefill 和 Decode阶段区别对待，61 层 Transformer (1–3 Dense, 4–61 MoE)恒定不变，仅仅在阶段决定如何部署(EP/DP专家部署, GPU 数量，冗余程度、跨卡通信模式）有所不同。



- 无论 Prefill 还是 Decode 阶段，都采用统一的 61 层 Transformer 结构，无需额外特殊处理。
- 每层 Transformer 中都内置了 Mixture-of-Experts（MoE）模块，用于高效稀疏激活专家，降低计算开销同时提高扩展能力。





## DeepSeek R1 模型的 MoE 架构与部署策略

DeepSeek R1 每层 Transformer decoder 由以下专家结构构成：

- 256 个路由专家+1 个共享专家：
  - 每个 Token 实际只通过门控路由激活 8 个路由专家，以及固定的 1 个共享专家，即每 Token 实际参与计算的专家数量共为 9 个。

### 高并发分布式部署方案

- Prefill 阶段：EP32 / DP32
  - 部署于 32 张 GPU（4节点 × 8 GPUs）
  - 路由专家冗余部署：每张 GPU 托管 9 个专家，共 288 个专家副本（其中32 个为冗余）
  - 每张 GPU 上额外部署了 1 个共享专家副本（数据并行）
- Decode 阶段：EP144 / DP144
  - 部署于 144 张 GPU（18节点 × 8 GPUs）
  - 路由专家冗余部署：每 GPU 存放 2 个专家，共 288 个专家副本（32个冗余）
  - 同样执行共享专家数据并行复制至每张 GPU

### 激活专家与冗余专家间的关系：

- 每个 Token 会依据门控得分从所有 GPU 上的专家副本中跨节点选出 8 个最佳路由专家，这种分布式路由能够最大化利用计算冗余，提高负载均衡效率与吞吐性能。



整体逻辑示意图：

![images](https://github.com/xinyuwei-david/david-share/blob/master/Deep-Learning/DeepEP/images/3.png)

需要注意的是，模型中两类Head的区别。

1. Multi-Head Attention 里的 “head”

• 出现在 **每一个 Transformer 层的 Attention 子模块** 中。
• 举例：DeepSeek-V3 的自注意力有 128 个 head，它们并行计算不同的注意力子空间，然后再拼在一起。
• 作用：帮助模型从多角度捕捉 token 之间的关系。
• 属于“层内部结构”，每一层都有。

2. LM Head（Language-Model Head）

• 只在 **整条网络的最末端出现一次**。
• 本质就是一个线性投影矩阵：`W ∈ ℝ^{d_model × vocab_size}`
– 把隐藏维度 d=7168 的向量，映射成词表大小 V=129 280 的 logits。
• 作用：把最后得到的语义表示转成“下一个词是谁”的概率分布。
• 通常与 Embedding 权重共享（weight tying），所以天然放在最尾。

简单对照

| 名称           | 位置                      | 数量            | 主要作用                        |
| -------------- | ------------------------- | --------------- | ------------------------------- |
| Attention head | 每层的 Attention 子模块内 | 每层几十~上百个 | 计算多视角注意力                |
| LM Head        | 整个网络结束后            | 仅 1 个         | 把最终隐藏向量投影成词表 logits |

因此：

• “每一层都有 head” 指的是 **Attention head**；
• **LM Head 只有最后一层才需要**，因为只有那时才要输出词概率。
把 LM Head 提前放到中间层既不符合语义，也会破坏模型的梯度与残差结构。

一句话记住

> Attention head ≠ LM Head——前者是层内并行小通道，后者是全网唯一的输出投影。



1个 **MoE Transformer** **Block**的内部实现示意：

![images](https://github.com/xinyuwei-david/david-share/blob/master/Deep-Learning/DeepEP/images/1.png)

1. 图里的 T1 / T2 / T3 / T4
   • T1 (Attention) → 这一层的自注意力子模块
   • T2 (Dispatch) → Router + 跨 GPU 发送 token bucket
   • T3 (Expert)  → 被选中专家执行 Linear-GELU-Linear
   • T4 (Combine) → 跨 GPU 把专家输出拉回原 GPU

###### Note:

```
*在 DeepSeek-V3 中*
*• 第 1–3 层是纯 Dense Block，没有 T2/T3/T4。*
*• **从第 4 层开始的 58 个 MoE Block**（4 ~ 61）内部，都可以套用这张 NVIDIA 图的 T1–T4 流水。*
*• 所以这张图“起点”相当于模型进入 **第 4 层** 之后的 任一 MoE 层，并一直重复到第 61 层结束。*
```

DeepEP 把重点火力全部集中在 **T2 Dispatch** 和 **T4 Combine** 这两段跨 GPU 通信环节，围绕“更少数据、更快路径、更好重叠”做了系统级改造；T1（Attention）与 T3（专家 MLP）几乎不改动，仅通过流重叠顺带获益。

下面分三栏把DeepEP优化点对照列清楚：

| T-段            | 传统做法 (NCCL All-to-All)                                   | DeepEP 的做法                                                | 收益                                                    |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------- |
| T1 Attention    | 本地稠密计算                                                 | **不改**，但通信后台化后 → 计算可与 T2/T4 真正并行           | GPU SM 利用率↑                                          |
| **T2 Dispatch** | • 每层一次 NCCL all_to_all<br>• 先单独 AllGather 头信息<br>• 统一 FP16/BF16 传输 | ① **精确分桶**：Router 只挑“跨卡 token”<br>② **FP8 压缩**：链路带宽占用 ↓ 60-70 %<br>③ **单核多流 DMA**：在同一 kernel 内并行发 NVLink / RDMA P2P<br>④ **Piggy-back header**：把路由头塞进 payload 首 64 B，省掉一次 AllGather | • NVLink/RDMA 实测带宽跑满<br>• 纯发送延迟 < 0.07 ms    |
| T3 Expert MLP   | GPU 本地计算，与通信串行                                     | **不改算法**，但：<br>• 通信核放低优先级 stream → Expert GEMM 与 T2/T4 完全重叠<br>• 可选 SM 配额：通信核仅占 0-24 SM，其余留给 GEMM | Expert 计算时间≈原来，但被前后通信完全掩蔽              |
| **T4 Combine**  | 再跑一次 all_to_all 回收输出                                 | ① **RDMA / NVLink 后台直写**：对端 HCA doorbell 立即 DMA<br>② **Low-lat kernel（解码）**：单 token 即发包，零批量聚合<br>③ **Handle Hook**：只在需要结果前才真正读显存 | • 纯回收延迟 < 0.08 ms<br>• 单 token P99 延迟 ↓ 40-60 % |

**DeepEP 做了什么？**

1. **通信路径改造**：All-to-All → 同步内核里并发触发多条 **P2P DMA**（NVLink + RDMA）。
2. **数据量削减**：只发送跨卡 token，且链路压成 **FP8**；头部随包携带，省一次 AllGather。
3. **极致重叠**：通信核放到低优先级 CUDA stream，配合 Hook + SM 配额，让 T2/T4 完全隐藏在 T1/T3 计算后面。
4. **两套内核**：
   • **normal** —— Prefill & 训练批量大，聚合发包，极限带宽；
   • **low-latency** —— 单 token 解码，即时 doorbell，极限时延。

**结果就是：**
• T2+T4 原本占 60% 以上的瓶颈被压缩到 < 15 %；
• GPU 利用率从 35 % 升至 95 % ；
• 训练吞吐提升 1.5-1.6×，解码单 token 延迟降到 ~0.4 ms。

**DeepEP 把 T2 / T4 这两段“大喇叭”通信，改造成窄而快的“直通水管”，让整条 T1→T4 流水线重新以计算为主而不是等网为主。**



## 为何 MoE 必须率先解决通信

1. **激活稀疏**
   token 仅走 k 个 Expert（k ≪ N_expert）→ 计算省，但 *路由信息* 激增。

2. **All-to-All 两跳**

   ```
   Router
     ├─ Dispatch : 把 token 送到目标 GPU / Expert
     ├─ Expert   : 多 GPU 并行 MLP
     └─ Combine  : 把结果拉回原 GPU
   ```

   

3. **通信成本基准**（Megatron-MoE, H100×8, 8 Expert, BS 4096）

   | 阶段     | NCCL All-to-All | 占总时长 |
   | -------- | --------------- | -------- |
   | Dispatch | 0.42 ms         | 35 %     |
   | Expert   | 0.48 ms         | 40 %     |
   | Combine  | 0.39 ms         | 25 %     |

   → **65 %** 的 step 时间在 *等网*，GPU-util ≈ 35 %。

------

## DeepEP = 专为 Expert-Parallel 打造的通信栈

### 设计目标

| 目标          | 可量化指标                              |
| ------------- | --------------------------------------- |
| 吃满带宽      | NVLink ≥ 150 GB/s；RDMA ≥ 35 GB/s       |
| 最小延迟      | 自回归单 token < 0.4 ms                 |
| 计算-通信重叠 | SM Idle < 5 %                           |
| 精准搬运      | 仅发跨卡 token；FP8 压缩后数据 ↓ ≥ 60 % |
| 易用性        | Python API；handle 自动管理             |

### 两套内核

| 内核            | 主攻方向 | 推荐场景       | 关键特性                     |
| --------------- | -------- | -------------- | ---------------------------- |
| **normal**      | 吞吐     | 训练 / Prefill | 批量聚合；Ring doorbell      |
| **low-latency** | 时延     | 自回归 Decode  | 单包即时 doorbell；SM 占用≈0 |

------

## 架构全貌

```
┌────────────── Python Layer ──────────────┐
│ Router (Top-k)                          │
│ Buffer API (dispatch / combine)         │
│  ├─ get_dispatch_layout()               │
│  └─ low_latency_dispatch()              │
└─▼───────── runtime.cu ── stream & event 管理 ──▲
   │
   ├─ NVLink P2P  (intranode.cu)  ── 自定义 PTX + GPU IPC
   │     • normal & low-lat kernels
   └─ RDMA  P2P  (internode.cu)  ── NVSHMEM_put / get
         • 每 Expert = 一组 QP,  VL0/1 分流
```

T1 / T2 / T3 / T4” 是我们在方案文档里用来描述 **通信距离（或代价）等级** 的抽象分层；
而 “NVLink P2P” 与 “RDMA P2P” 则是落地实现时调用的 **具体链路 & API**。
两套命名可以一一对应——见下表：

| 逻辑层级 | 典型距离 / 代价                                              | 主要链路 & API                                            | 在代码中的实现文件 | 对应关系             |
| -------- | ------------------------------------------------------------ | --------------------------------------------------------- | ------------------ | -------------------- |
| **T1**   | GPU **片内**<br>• 同 SM warp-shuffle / shared-mem<br>• 同卡不同 SM 的 L2 copy | CUDA `__shfl_*`, `memcpy_async`, NCCL P2P within device   | 内核代码自身       | （无外链路，仅片内） |
| **T2**   | **同节点 GPU↔GPU**（机箱里）<br>NVLink 交换矩阵              | NVLink **P2P**<br>自定义 `PTX` / `cudaMemcpyAsync` / NCCL | `intranode.cu`     | NVLink P2P = **T2**  |
| **T3**   | **跨节点**、同机房 TOR 内<br>IB/NDR/HDR 网                   | **RDMA P2P**<br>`NVSHMEM_put/get` 或 ibverbs QP           | `internode.cu`     | RDMA P2P = **T3**    |
| **T4**   | 更远（跨机房 / 跨 Region）<br>需要多跳交换                   | TCP + gRPC / RoCE + RAAI<br>或分层 AllReduce / 参数服务器 | 上层框架（非本库） | T4 需额外组件        |

要点说明

1. T1–T4 是“抽象层级”；NVLink P2P、RDMA P2P 只是其中两级（T2、T3）的**具体实现**。
2. runtime.cu 会根据 `src_gpu` 与 `dst_gpu` 的拓扑关系决定走哪一级：
   • 同主机 → 调 `intranode.cu` (T2)
   • 跨主机 → 调 `internode.cu` (T3)
3. 若训练规模只在单机内，通信永远停留在 T1+T2；集群放大到多机时才会上升到 T3/T4。

因此，“T1 T2 T3 T4” 给的是**抽象概念与性能期望**；
“NVLink P2P / RDMA P2P” 给的是**如何在代码里落地该层级**，两者并不冲突，而是层级→实现的映射关系。

### 数据流 12 步

| #    | Step（操作）                                        | Device / Link      | Purpose / 作用                          | Typical Latency (A100, ≈8 k token) | T-Stage                                       |
| ---- | --------------------------------------------------- | ------------------ | --------------------------------------- | ---------------------------------- | --------------------------------------------- |
| 1    | Kernel 扫 top-k，判定本卡 / 跨卡 token              | GPU                | 找出需跨 GPU 发送的稀疏 token 子集      | 8–12 µs                            | **T1**                                        |
| 2    | `cub::Scan` 生成 prefix-sum offset 表               | GPU                | 计算每个 token 在 send-buf 中的写入偏移 | 0.02–0.08 ms                       | **T1**                                        |
| 3    | `cudaMallocAsync` 精确分配 send-buf（首帧 +3–6 µs） | GPU                | 按总字节数申请显存，避免过分配          | <0.01 ms                           | **T1**                                        |
| 4    | 打包 & FP8 压缩                                     | GPU                | 将跨卡 token 写入 send-buf 并降低位宽   | 0.03–0.10 ms                       | **T1**                                        |
| 5    | 触发 NVLink P2P / RDMA-GDR DMA                      | GPU→DMA            | 把压缩数据直推目标 GPU                  | NVLink≈0.03 ms<br>RDMA≈0.06 ms     | **T2** (同节点)<br>**T3** (跨节点) — Dispatch |
| 6    | Expert MLP 计算（FP8 GEMM）                         | GPU                | 对已路由到本卡的 token 执行前向 GEMM    | 0.20–0.40 ms                       | **T1**                                        |
| 7    | Doorbell 写 flag（NVSHMEM）                         | HCA                | GPU DMA 在对端 QP 置位，通知数据就绪    | overlapped in #6                   | **T2/T3** — Dispatch 末尾                     |
| 8    | Combine 端读取 handle / 偏移                        | GPU                | 读取对端 flag，拿到 recv-buf 基址与长度 | 0.02 ms                            | **T4** — Combine 起点                         |
| 9    | 稀疏 axpy 加权求和                                  | GPU                | 将跨卡结果与本卡结果按权重合并          | 0.05–0.20 ms                       | **T4** — Combine                              |
| 10   | 反向阶段：dispatch / combine 对调                   | GPU                | 在 BP 中执行相同 10 步但方向相反        | 同上                               | Dispatch = **T2/T3**；Combine = **T4**        |
| 11   | Checkpoint Push（可选）                             | GPU → CPU → S3/GCS | 周期性将模型快照写入对象存储            | 1–3 s / GB                         | **T4** (跨 DC，低频)                          |
| 12   | 异地同步 / 灾备复制（可选）                         | WAN TCP / gRPC     | 将最新 ckpt / 日志同步至异地 Region     | 100–300 ms RTT + size              | **T4** (跨 Region，超长距)                    |

说明

1. \#1–10 为 **高频核心路径**，覆盖一次 micro-batch 的 Dispatch 与 Combine，并与示意图中的 T1–T4 一一对应。
2. \#11–12 为 **低频扩展步骤**，仅在需要跨 Region 容灾、冷备或多活部署时启用；不会影响 #1–10 的亚毫秒级性能。
3. 若训练仅限单节点，可忽略 RDMA 分支（T3）与扩展步骤；若需更细的内核级调优，可在 Step 1、4、6 内再做 profiler 级拆分。

#### 内核亮点速览

| 关键词       | 说明                                                         |
| ------------ | ------------------------------------------------------------ |
| PTX 指令     | `ld.global.nc.L1::no_allocate.L2::256B` —— 不污染 L1，256 B 对齐直写 L2→NVLink |
| RDMA 直写    | `nvshmemx_putmem_nbi_block()`；GPU 侧发 WR，无 CPU 轮询      |
| Hook-overlap | `hook()` 在需要结果前才触发 RDMA read，实现双 micro-batch overlap |
| SM 配额      | `Buffer.set_num_sms(24)` → 通信 kernel 仅占 1/6 SM，其余给 GEMM |

------

## 编程模型示例

> 目标：在 **H100/H800 + CUDA ≥ 12.3 + PyTorch ≥ 2.1** 环境下，
> 编译 NVSHMEM → 安装 DeepEP Python 扩展 → 跑通最小示例。
> ​*Ampere (A100) 亦可运行，但 FP8 会自动回退为 FP16，带宽/显存收益约降 20 %。*

### 环境

| 组件          | 版本要求            | 入手方式                         |
| ------------- | ------------------- | -------------------------------- |
| NVIDIA Driver | ≥ 535               | `ubuntu-drivers` 或 NVIDIA 官网  |
| CUDA Toolkit  | ≥ 12.3              | `sudo apt install cuda-12-3`     |
| NCCL          | 与 CUDA 同版        | 随 Toolkit 安装                  |
| NVSHMEM       | **v2.10+** (需 GDR) | 源码编译，见下文                 |
| PyTorch       | ≥ 2.1 (cu12.x)      | `pip install torch==2.1.0+cu121` |
| Python        | 3.8 – 3.11          | 官档或 Conda                     |

### 编译并安装 NVSHMEM

```
# ① 克隆源码
git clone --depth=1 https://github.com/NVIDIA/nvshmem.git
cd nvshmem

# ② 配置 & 编译（启用 GPU-Direct RDMA + GDRCopy）
make -j $(nproc) CUDA_HOME=/usr/local/cuda PREFIX=$HOME/nvshmem \
     NVSHMEM_BUILD_GDRCOPY=1

# ③ 安装到自定义前缀
make install PREFIX=$HOME/nvshmem

# ④ 告知系统库搜索路径
echo 'export NVSHMEM_DIR=$HOME/nvshmem' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$NVSHMEM_DIR/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

检查安装：

```
ls $NVSHMEM_DIR/lib | grep libnvshmem   # 应看到 libnvshmem_host.so 等文件
```

### 获取并安装 DeepEP

> 假设公司/团队把 DeepEP 托管在 `<org>/deep_ep`；
> 若已发布到 PyPI，请将第 2 步替换为 `pip install deepep`.

```
# ① 克隆仓库
git clone git@github.com:<org>/deep_ep.git
cd deep_ep

# ② 本地开发模式安装（便于调试）
pip install -e .        # 等同于 python setup.py develop
#   或
# pip install .         # 生产环境用正式 install

# ③ 验证 Python 扩展能正确加载 NVSHMEM
python - <<'PY'
import importlib, torch
deep_ep = importlib.import_module("deep_ep")
print("DeepEP version:", deep_ep.__version__)
buf = deep_ep.Buffer(torch.distributed.group.WORLD)  # 不报错即成功
PY
```

### 最小运行示例（单机 8 GPU）

```
#!/usr/bin/env bash
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
WORLD=8

python -m torch.distributed.run \
       --nproc_per_node=$WORLD \
       demo_moe.py               # deep_ep/examples/demo_moe.py
```

`demo_moe.py` 核心片段（FP8 正向 / FP16 反向）：

```
from deep_ep import Buffer, set_global_expert_mlps
import torch, torch.distributed as dist

group = dist.new_group()
buf   = Buffer(group, nvl_bytes=0, rdma_bytes=256<<20)
Buffer.set_num_sms(24)                     # 通信 kernel 占 1/6 SM

def forward(x_fp8, topk_idx):
    recv, hdl, hook = buf.dispatch(x_fp8, topk_idx, num_experts=8)
    y = expert_mlp(recv)                   # 自定义 MLP
    out, _        = buf.combine(y, hdl)
    return out
```

### 常见问题排查

| 现象                                                 | 可能原因                                    | 解决方案                                                  |
| ---------------------------------------------------- | ------------------------------------------- | --------------------------------------------------------- |
| `libnvshmem_host.so: cannot open shared object file` | `LD_LIBRARY_PATH` 未包含 `$NVSHMEM_DIR/lib` | 重新 `source ~/.bashrc`                                   |
| `RuntimeError: FP8 tensor core not supported`        | GPU 为 A100/Ampere                          | 设置 `export DEEP_EP_FORCE_FP16=1` 或升级至 H100          |
| RDMA 吞吐仅 20 GB/s                                  | GDRCopy 未开启 / NIC 不支持 NDR             | 确认 `make` 时 `NVSHMEM_BUILD_GDRCOPY=1`，并升级 HCA 固件 |

### 一键 Docker（可选）

```
FROM nvidia/cuda:12.3.2-devel-ubuntu22.04
RUN apt-get update && apt-get install -y git build-essential python3-pip \
    && pip install torch==2.1.0+cu121 \
    && git clone --depth=1 https://github.com/NVIDIA/nvshmem.git \
    && cd nvshmem && make -j$(nproc) CUDA_HOME=/usr/local/cuda \
         PREFIX=/opt/nvshmem NVSHMEM_BUILD_GDRCOPY=1 \
    && make install PREFIX=/opt/nvshmem \
    && echo "export NVSHMEM_DIR=/opt/nvshmem" >> /etc/profile \
    && echo "export LD_LIBRARY_PATH=\$NVSHMEM_DIR/lib:\$LD_LIBRARY_PATH" >> /etc/profile \
    && git clone https://github.com/<org>/deep_ep.git && cd deep_ep \
    && pip install .
```

完成以上步骤，即可在 **H100/H800**（或 A100、功能降级）上直接调用 `deep_ep.Buffer`、体验“FP8 正向 + FP16 反向” 的 Expert-Parallel 加速效果。



## 性能实测

### 训练吞吐（train-tokens / s / GPU，top-8）

| GPU×Node | Expert | NCCL   | DeepEP     | 提升    |
| -------- | ------ | ------ | ---------- | ------- |
| 8 × 1    | 8      | 190 k  | **300 k**  | ↑ 1.58× |
| 64 × 8   | 64     | 1.46 M | **2.33 M** | ↑ 1.60× |

### 单 token 延迟（Decode，batch = 128）

| Expert | NCCL (ms) | DeepEP-normal | DeepEP-LL | 说明        |
| ------ | --------- | ------------- | --------- | ----------- |
| 8      | 0.78      | 0.52          | **0.38**  | NVLink 内   |
| 256    | 1.23      | 0.74          | **0.45**  | 跨机 + RDMA |

### 火焰图对比（8 GPU）

```
NCCL AllToAll        DeepEP normal
██████ 通信          █ 通信提交
▒▒▒▒▒▒ 计算          ████ 计算
██████ 通信          ░░ 通信后台
```

GPU Idle: **39 % → 4 %**



**Refer to：**

https://github.com/naklecha/llama3-from-scratch?tab=readme-ov-file