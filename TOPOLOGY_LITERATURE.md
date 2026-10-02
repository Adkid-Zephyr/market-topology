# 本轮采用的文献与证据范围

2026-09-28 核查。筛选标准是原始论文、明确的方法、可核对的数学或实验依据与清楚的适用范围。发表地点是筛选线索，不是预测能力的保证。没有将搜索引擎摘要或宣传性表述当成已证明的危机预测结果。

| 文献 | 可核查来源 | 本轮采用什么 | 不能据此声称什么 |
|---|---|---|---|
| Bubenik, 2015, Statistical Topological Data Analysis using Persistence Landscapes | [JMLR 16](https://jmlr.org/papers/v16/bubenik15a.html) | 景观函数的稳定表示与统计学习接口；保留曲线，不只压成范数 | 稳定性定理不等于金融预测有效；序列相关性还需另外处理 |
| Carrière, Cuturi & Oudot, 2017, Sliced Wasserstein Kernel for Persistence Diagrams | [ICML / PMLR 70](https://proceedings.mlr.press/v70/carriere17a.html)，全文 Definition 3.1、Algorithm 1 | 完整有限持续图之间的 sliced-Wasserstein 距离及正定核；本轮 16 方向近似 | 原文非金融任务上的效果，不能直接转成危机预测效果 |
| Adams et al., 2017, Persistence Images | [JMLR 18](https://jmlr.org/papers/v18/16-337.html) | 支持对持续图使用有理论依据的向量表示这一方向，列为后续备选 | 本轮没有另实现 persistence images，不把它的结果算入比较 |
| Gidea et al., 2020, Topological recognition of critical transitions in time series of cryptocurrencies | [出版社](https://www.sciencedirect.com/science/article/pii/S0378437119321363)，[作者所在大学存档](https://upcommons.upc.edu/entities/publication/a66a8df7-7d02-46ef-b4bd-76cea6519b41) | 延迟嵌入的拓扑路线；借鉴表示，不复刻事后分段及聚类结果 | 选定加密货币事件的识别不能代替长历史样本外评估 |
| Katz & Biem, 2021, Time-resolved topological data analysis of market instabilities | [出版社](https://www.sciencedirect.com/science/article/pii/S0378437121000881) | 时间分辨 TDA、延迟嵌入，以及扩展到个股/CDS 的思路 | 其 200 家企业股价/CDS 数据没有在本项目取得；这里不是该完整实验的复现。作者也说明外生冲击及精确时点预测的局限 |
| Akingbade et al., Why Topological Data Analysis Detects Financial Bubbles? | [作者原稿](https://arxiv.org/abs/2304.06877) | LPPLS 条件下振荡结构与拓扑信号联系的解释 | 以特定泡沫模型为前提，不是所有危机前必有拓扑环的普遍定理 |
| Akingbade, 2026, Null-Validated Topological Signatures of Financial Market Dynamics | [arXiv v2](https://arxiv.org/html/2602.00383v2) | 近期关于替代数据检验和额外时间结构的研究线索；前轮已测试 Gaussian 对照 | 本轮依据为 arXiv 文本，未另核对其正式发表版本；作者仍把预测价值列为待研究方向 |

原始基线仍来自 [Gidea & Katz 2018](https://arxiv.org/html/1703.04385)。以上文献用于确定本轮方法与限制，不能合并成“文献已经证明可以预测金融危机”的结论。

## 本轮实现与原文的关系

- 景观模型是有限网格/层数的工程实现：5 层、64 点网格、训练集 PCA8，不是保留全部连续函数。
- 持续图模型保留全部有限 H0/H1 条，采用 16 个固定方向近似原文 Algorithm 1，并用核岭回归而非其分类实验中的 SVM。加入训练标签均值作为基准，输出截断至 [0,1]。
- 延迟模型采用固定 4 维、tau=1/5、点云 50/100；不是把原论文在比特币或企业 CDS 上的最佳参数搬到美国股指。
- 所有参数选择在嵌套时间验证的内层完成。文献中适用于独立样本的随机划分没有直接照搬到重叠的金融预测标签上。
