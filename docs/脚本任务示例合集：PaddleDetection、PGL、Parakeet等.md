> 更新说明：
> 2020.09.26 新增PaddleDetection脚本任务示例：[PaddleDetection：PCB瑕疵检测](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1029223)，暂时只跑通P40单卡训练
# 前言 

- 如果任务要训练很久如何不掉线？
- 能不能同时开多个项目训练？
- 想试用多卡训练训练的功能却不知道脚本项目怎么入门？

使用AI Studio平台时，绝大部分时间我们都在Notebook环境下开发，但是Notebook项目有数量和离线时间限制。

<font size=5> 本文整理了一些适合使用脚本任务训练的示例，同时提供了Session-based Recommendation SR-GNN在自定义数据集上脚本任务简单使用说明，后续会持续补充各场景下更多脚本任务示例。</font>

<font color=red size=4 >如果您想使用自己业务场景下的数据，只要令数据满足对应脚本任务数据格式要求即可开始训练。</font>


# 脚本任务结构解读
![file](https://ai-studio-static-online.cdn.bcebos.com/c93f730754234f0a8b5e68ddf669817393d1722a4c524d77b1f8234e44282f16)

<font size=5>STEP 1 模拟终端</font>

由于脚本任务没有终端，因此将标记为【main】的`train.py`改名为`run.py`，起到和终端输入命令相同的功能。

<font size=5>STEP 2 指定百度源</font>

从日志看脚本任务内置的镜像源存在问题，因此需要手动指定百度源等国内镜像，下载好必须的依赖。

<font size=5>STEP 3 安装正确的飞桨版本</font>

脚本任务内置的框架较少，一些任务需要用到1.6、1.7等，建议直接`pip`安装对应版本。

<font size=5>STEP 4 准备数据</font>

1. 直接上传：脚本任务可以上传<30M的数据。
2. 数据集挂载：超过30M的数据集，可以先上传到数据集中再挂载，注意设置正确的路径。

<font size=5>STEP 5 训练并下载输出</font>

1. 在`run.py`中指定要运行的python文件开始训练。
2. 训练完成后，将需要下载的文件移动到输出目录，待脚本任务完成即可下载。

## $\color{red}{一些注意事项}$
1. 脚本任务不能创建空目录，即使创建了，运行时也会消失。
2. 脚本任务可能会出现I/O报错，一些对策如下：
-  如果需要从数据集解压缩文件到data等指定目录，该data目录需要是在`run.py`中用`mkdir`命令创建的
-  尽量等训练全部完成再将需要下载的文件移动到下载输出目录
3.脚本任务目前尚不能解析YAML和Markdown等文件格式，但是各有个小技巧，将文件后缀改成.py就可以查看内容了

# PaddleDetection脚本任务示例

PaddleDetection飞桨目标检测开发套件，旨在帮助开发者更快更好地完成检测模型的组建、训练、优化及部署等全开发流程。

PaddleDetection模块化地实现了多种主流目标检测算法，提供了丰富的数据增强策略、网络模块组件（如骨干网络）、损失函数等，并集成了模型压缩和跨平台高性能部署能力。

经过长时间产业实践打磨，PaddleDetection已拥有顺畅、卓越的使用体验，被工业质检、遥感图像检测、无人巡检、新零售、互联网、科研等十多个行业的开发者广泛应用。

## 使用说明

<font size=4>1. PaddleDetection的脚本任务情况相对特殊，需要先通过数据集挂载加载好PaddleDetecion源码，再解压缩。</font>
![file](https://ai-studio-static-online.cdn.bcebos.com/be3096d2f0df458590a981ae0372964b32cc0d8f4af449b5a956d404a7c91abf)

<font size=4 color=red>2. 对于PaddleDetection训练需要指定的自定义config文件，需要在本地调好，上传到编辑界面，然后指定绝对路径开始训练</font>
![file](https://ai-studio-static-online.cdn.bcebos.com/cf7b5074e58047a7a4f436af6737d8b5f2fd07ceb8624b02a784a3adde393e78)

## [PaddleDetection：PCB瑕疵检测](https://aistudio.baidu.com/aistudio/clusterprojectdetail/1029223)

# PGL：图学习脚本任务示例合集

Paddle Graph Learning (PGL)是一个基于PaddlePaddle的高效易用的图学习框架，本项目中的示例整理自文档的[examples目录](https://github.com/paddlepaddle/PGL/tree/master/examples)，均已测试跑通。

关于PGL的入门教程，[自尊心3](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/218586)大佬的[PGL：Paddle带你走进图学习](https://aistudio.baidu.com/aistudio/projectdetail/413386)系列详细代码解读一定不可错过。

![file](https://gitee.com/paddlepaddle/PGL/raw/master/docs/source/_static/framework_of_pgl.png)

## [GCN：图卷积网络](https://aistudio.baidu.com/aistudio/clusterprojectdetail/881609)
图卷积神经网络[Graph Convolutional Network (GCN)](https://arxiv.org/abs/1609.02907)是图学习中的一个基本的网络结构。

> ### 数据集
> 3个常用的图学习数据集，CORA, PUBMED, CITESEER。数据集的相关介绍参考[论文：Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)。

参考资料：
- [PGL系列4：Graph Convolutional Networks](https://aistudio.baidu.com/aistudio/projectdetail/409400)
- [何时能懂你的心——图卷积神经网络（GCN）](https://zhuanlan.zhihu.com/p/71200936)
## [GAT：基于Attention的图卷积网络](https://aistudio.baidu.com/aistudio/projectdetail/881633)

[Graph Attention Networks (GAT)](https://arxiv.org/abs/1710.10903)网络是GCN网络结构改进版本，通过学习自注意层，给节点分配权重，以解决图卷积或其近似方法的缺点。

> ### 数据集
> CORA, PUBMED, CITESEER。

参考资料：
- [PGL系列5：Graph Attention Networks](https://aistudio.baidu.com/aistudio/projectdetail/411016)

## [GraphSage：基于邻居采样的大规模图卷积网络](https://aistudio.baidu.com/aistudio/clusterprojectdetail/891560)

GCN要求在一个确定的图中去学习顶点的embedding，无法直接泛化到在训练过程没有出现过的顶点，即属于一种直推式(transductive)的学习。GraphSAGE则是一种能够利用顶点的属性信息高效产生未知顶点embedding的一种归纳式(inductive)学习的框架，其核心思想是通过学习一个对邻居顶点进行聚合表示的函数来产生目标顶点的embedding向量。

> ### 数据集Reddit Dataset
> 
> 使用的是Reddit数据集。这个数据集包含了不同Reddit帖子和所属的社区，用来对帖子所属社区进行分类的任务。Reddit是一个大型在线讨论论坛，用户可以在其中发布和评论不同主题社区中的内容。这个数据集手机了2014年9月发布的帖子，构建了一个图形数据集。在这个数据集中，节点标签是在帖子所属的社区或者“subreddit”。数据集对50个大型社区进行了抽样，并构建了一个帖子到帖子的连接图表。如果一个用户在两个帖子上都发表评论，则将这两个帖子连接起来。该数据及总共包含了232965个帖子，每个节点的平均度为294.

参考资料：
- [【Graph Neural Network】GraphSAGE: 算法原理，实现和应用](https://zhuanlan.zhihu.com/p/79637787)
- [PGL系列8：GraphSAGE](https://aistudio.baidu.com/aistudio/projectdetail/519067)

## [unSup-GraphSage：无监督学习的GraphSAGE](https://aistudio.baidu.com/aistudio/projectdetail/891796)

GraphSAGE在无监督学习任务上的应用。

## [LINE：基于一阶、二阶邻居的表示学习](https://aistudio.baidu.com/aistudio/projectdetail/891878)

LINE: Large-scale Information Network Embedding主要解决了**大规模**网络嵌入到低维向量。而低维向量能够更加有助于visualization, node classification, 和link prediction。这个方法能够处理各种各样的网络，有向无向图，带不带权等等。论文作者认为文章主要有如下两个亮点：

- 优化的目标函数设计的比较好
- 边采样方法能够克服(address)传统的随机梯度下降方法的缺陷，同时提高最后结果的效率和效果（efficiency and effectiveness）。

### 数据集

> LINE案例中使用的数据集是[Flickr社交网络](http://socialnetworks.mpi-sws.org/data-imc2007.html)，该数据集是论文[Measurement and Analysis of Online Social Networks Systems](https://link.springer.com/referenceworkentry/10.1007/978-1-4939-7131-2_242)的发布的三个数据集之一，在会议Internet Measurement Conference2017中发表。
> 
> Flickr是雅虎旗下图片分享网站，为一家提供免费及付费数位照片储存、分享方案之线上服务，也提供网络社群服务的平台。其重要特点就是基于社会网络的人际关系的拓展与内容的组织。这个网站的功能之强大，已超出了一般的图片服务，比如图片服务、联系人服务、组群服务。更多可以查看[百度百科flickr](https://baike.baidu.com/item/flickr/714868)
> 
> 在这个Flickr数据集中，包含 1,715,256个节点，22,613,981条边（无向且不加权）。节点的平均度 26.37。
> 
> 所有节点分为5个类别（组），每个节点包含一个或多个标签，既每个节点可能包含于一个或多个组。

参考资料：
- [LINE:Large-scale Information Network Embedding阅读笔记](https://zhuanlan.zhihu.com/p/27037042)  
- [LINE：大规模信息网络嵌入](https://www.shintaku.top/posts/line/)  
- [【论文笔记】LINE：大规模信息网络嵌入](https://blog.csdn.net/wizardforcel/article/details/95016540)  
- [LINE：Large-scale Information Network Embedding翻译](https://www.cnblogs.com/bianque/articles/10771029.html)
- [PGL系列12：LINE大型信息网络嵌入](https://aistudio.baidu.com/aistudio/projectdetail/655814)

## [DeepWalK：DFS随机游走的表示学习](https://aistudio.baidu.com/aistudio/clusterprojectdetail/895869)

Deepwalk是一种将随机游走(random walk)和word2vec两种算法相结合的图结构数据挖掘算法。该算法主要分为随机游走和生成表示向量两个部分。首先利用随机游走算法(Random walk)从图中提取一些顶点序列；然后借助自然语言处理的思路，将生成的定点序列看作由单词组成的句子，所有的序列可以看作一个大的语料库(corpus)，最有利用自然语言处理工具word2vec将每一个顶点表示为一个维度为d的向量。

### 数据集
> BlogCatalog数据集是一个社会关系网络，图是由博主和他（她）的社会关系（比如好友）组成，labels是博主的兴趣爱好。Reddit数据集是由来自Reddit论坛的帖子组成，如果两个帖子被同一人评论，那么在构图的时候，就认为这两个帖子是相关联的，labels就是每个帖子对应的社区分类。Epinions是一个从一个在线商品评论网站收集的多图数据集，里面包含了多种关系，比如评论者对于另一个评论者的态度（信任/不信任），以及评论者对商品的评级。
> 
> 文件构成
> BlogCatalog数据集的结点数为10312，边条数为333983，label维度为39，数据集包含两个文件：
> - Nodes.csv：以字典的形式存储用户的信息，但是只包含节点id。
> - Edges.csv：存储博主的社交网络（好友等），以此来构图。

参考资料：
感谢[没入门的研究生](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/300157)分享的论文解读：
- [DeepWalk：火爆的图神经网络前夜](https://aistudio.baidu.com/aistudio/projectdetail/532867)
- [DeepWalk（续）：如何处理大型网络](https://aistudio.baidu.com/aistudio/projectdetail/534062)

## [MetaPath2Vec：基于metapath的表示学习](https://aistudio.baidu.com/aistudio/projectdetail/897008)

MetaPath2Vec: Scalable Representation Learning for Heterogeneous Networks研究的是关于异构网络的特征表示学习。异构网络的最大挑战来源于不同种类的节点与连接，因此限制了传统network embedding的可行性。论文提出了两种特征学习模型：metapath2vec以及metapath2vec++，它们的具体做法是基于元路径的随机游走来指定一个节点的邻居，之后利用异构skip-gram模型来实现embedding。

### 数据集
> DBLP——Digital Bibliography & Library Project的缩写。这里是[DBLP的主页](http://www.informatik.uni-trier.de/~ley/db/)
> 
> DBLP是计算机领域内对研究的成果以作者为核心的一个计算机类英文文献的集成数据库系统，按年代列出了作者的科研成果。包括国际期刊和会议等公开发表的论文。DBLP没有提供对中文文献的收录和检索功能，国内类似的权威期刊及重要会议论文集成检索系统有C-DBLP。
> 
> 这个项目是德国特里尔大学的Michael Ley负责开发和维护。它提供计算机领域科学文献的搜索服务，但只储存这些文献的相关元数据，如标题，作者，发表日期等。和一般流行的情况不同，DBLP并没有使用数据库而是使用XML存储元数据。
> 
> 示例项目使用了其中部分数据。

参考资料：
- [论文链接](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf)
- [metapath2vec: Scalable Representation Learning for Heterogeneous Networks 阅读笔记](https://zhuanlan.zhihu.com/p/32598703)
## [Node2Vec：结合DFS及BFS的表示学习](https://aistudio.baidu.com/aistudio/projectdetail/897176)

Node2vec是用来产生网络中节点向量的模型，输入是网络结构（可以无权重），输出是每个节点的向量。

### 数据集
> Node2Vec示例中提供了BlogCatalog和Arxiv两种数据集的实现。

参考资料：
- [PGL系列7：node2vec](https://aistudio.baidu.com/aistudio/projectdetail/481549)

## [Struct2Vec：基于结构相似的表示学习](https://aistudio.baidu.com/aistudio/projectdetail/897467)

与很多现有的做网络表示的工作不同，Struct2Vec关注的是不同的节点在网络中的所处的角色。两个不近邻的顶点也可能拥有很高的相似性，可能这些节点在邻域中的角色相似，例如星型结构的中心节点，社区结构之间的桥接节点等。struc2vec就是针对捕捉节点的结构化角色相似度（structural role proximity）提出的模型。

> ### 数据集
> 
> 美国空中交通网络数据集 usa-airports.edgelist labels-usa-airports.txt 数据收集来自[Bureau of Transportation Statistics](https://www.transtats.bts.gov/)，记录了美国2016年一月和十月之间的航班活动情况。网络数据有1190个节点，13599个边。机场的繁忙程度（label）使用机场的出入人数，按照排序分级。

参考资料：
- [【论文笔记】struc2vec](https://zhuanlan.zhihu.com/p/63175042)
- [PGL系列14：Struc2Vec](https://aistudio.baidu.com/aistudio/projectdetail/897514)

## [SGC：简化的图卷积网络](https://aistudio.baidu.com/aistudio/projectdetail/897570)

SGC(Simplifying Graph Convolutional Networks)是基于图卷积网络GCN简化而来。SGC的目的就是要把非线性的GCN转化成一个简单的线性模型，通过消除GCN层之间的非线性操作和缩小权重矩阵，将得到的函数折叠成一个线性变换来减少GCNs的额外复杂度。论文证明了SGC相当于一个固定的低通道滤波器和一个线性分类器，SGC中的特征提取等价在每个特征的维度上应用了单个固定的filter。实验结果表明，这些简化不会对许多下游应用的准确性产生负面影响。此外，得到的SGC模型可扩展到更大的数据集，并且比FastGCN产生两个数量级的加速。

> ### 数据集
> CORA, PUBMED, CITESEER。

参考资料：
- [PGL系列13：SGC: 简化的GCNs](https://aistudio.baidu.com/aistudio/projectdetail/739861)

## [DGI：基于图卷积网络的无监督表示学习](https://aistudio.baidu.com/aistudio/clusterprojectdetail/898132)

Deep Graph Infomax (DGI)是一种通用的无监督图学习方法，它可以用于学习节点的嵌入向量。DGI通过最大化深度神经网络编码器的输入和输出之间的互信息，研究无监督表示学习。

> ### 数据集
> CORA, PUBMED, CITESEER。

参考资料：
- [PGL系列9： DGI: Deep Graph Infomax](https://aistudio.baidu.com/aistudio/projectdetail/548667)

## [GATNE：一种针对 Multi-Edge 的大规模异构图嵌入模型](https://aistudio.baidu.com/aistudio/clusterprojectdetail/872114)

清华大学和达摩院合作的一篇论文《Representation Learning for Attributed Multiplex Heterogeneous Network》，发表于 KDD 2019。

目前很多 Graph Embedding 应用广泛，但大部分都只是同构网络或者是小尺度网络，而真实世界往往大都是数以亿计的不同类型的节点和边，且节点往往包含多种属性。

为此，作者提出了 GATNE 框架用于解决大规模多元异构属性网络（Attributed Multiplex Heterogeneous Network，AMHEN），该框架支持 transductive 和 inductive 的学习范式。

此外，作者也进行了理论分析证明了 GATNE 具有良好的表达能力，并通过四种不同的数据集和 A/B 测试验证了模型的性能。

![https://ucc.alicdn.com/pic/developer-ecology/d1799b1f3ca74a3b9dbf8cb68c51d0ea.png](https://ucc.alicdn.com/pic/developer-ecology/d1799b1f3ca74a3b9dbf8cb68c51d0ea.png)

参考资料：
- [Representation Learning for Attributed Multiplex Heterogeneous Network](https://arxiv.org/pdf/1905.01669.pdf)  
- [KDD 2019 | GATNE：一种针对 Multi-Edge 的大规模异构图嵌入模型](https://developer.aliyun.com/article/714557)
- [PGL系列15：GATNE](https://aistudio.baidu.com/aistudio/projectdetail/978916)


# Parakeet：语音合成脚本任务示例
飞桨语音合成套件，提供了灵活、高效、先进的文本到语音合成工具，帮助开发者更便捷高效地完成语音合成模型的开发和应用。本项目中的示例整理自文档的[examples目录](https://github.com/PaddlePaddle/Parakeet/tree/develop/examples)，均已测试跑通。
## [脚本任务示例：Parakeet——WaveFlow语音合成模型训练](https://aistudio.baidu.com/aistudio/projectdetail/629531)
关于该项目的详细说明，可参考:
- [Parakeet：手把手教你训练语音合成模型（脚本任务、Notebook）](https://aistudio.baidu.com/aistudio/projectdetail/639029)
- [Parakeet：飞桨，你是个成熟的框架了，要学会自己读论文](Parakeet：飞桨，你是个成熟的框架了，要学会自己读论文)

# 推荐系统：[SR-GNN脚本任务示例应用](https://aistudio.baidu.com/aistudio/clusterprojectdetail/913252)

## 用户下个月会买什么？

> 该示例参考自：[基于PaddlePaddle的SR-GNN推荐算法](https://aistudio.baidu.com/aistudio/projectdetail/124382)

原项目中复现了论文效果，在DIGINETICA数据集上P@20可以达到50.7；本文则使用用户购买预测常规赛的数据集[MarTech Track1](https://aistudio.baidu.com/aistudio/datasetdetail/19383)，将其迁移为一个预测下月用户购买场景的任务，测试集上P@3可以达到24.3。

脚本任务项目中目录结构及说明：

```text
.
├── run.py            # 运行脚本
├── train.py             # 训练脚本
├── eval.py             # 验证脚本
├── infer.py             # 预测脚本
├── network.py           # 网络结构
├── reader.py            # 和读取数据相关的函数
├── data/martech
    ├── config.txt        # 用户购买商品数量
    ├── train.txt       # 生成的二进制训练集文件
    ├── test.txt       # 生成的二进制测试集文件
```

运行效果：

![file](https://ai-studio-static-online.cdn.bcebos.com/6944648abb58449c8ee0dc0e30d57d31aa447fcd2be443fdbce61ce9925c6509)

## 简介

SR-GNN模型的介绍可以参阅论文[Session-based Recommendation with Graph Neural Networks](https://arxiv.org/abs/1811.00855)。

本文解决的是Session-based Recommendation这一问题,过程大致分为以下四步：

是对所有的session序列通过有向图进行建模。

然后通过GNN，学习每个node（item）的隐向量表示

然后通过一个attention架构模型得到每个session的embedding

最后通过一个softmax层进行全表预测

模型结构如下图：
![file](https://ai-studio-static-online.cdn.bcebos.com/3395a79e26434c18bbe798bfe66c7a73b3df92d7bd9041f6a275ba5b8ebfa245)


SR-GNN最初处理的是用户点击session预测的问题，运行数据预处理之后在data文件夹下会产生diginetica文件夹，里面包含config.txt、test.txt、train.txt三个文件

生成的数据格式为:(session_list, label_list)。

其中session_list是一个session的列表，其中每个元素都是一个list，代表不同的session。label_list是一个列表，每个位置的元素是session_list中对应session label。

例子：session_list=[[1,2,3], [4], [7,9]]。代表这个session_list包含3个session，第一个session包含的item序列是1,2,3，第二个session只有1个item 4，第三个session包含的item序列是7，9。

label_list = [6, 9, 1]。代表[1,2,3]这个session的预测label值应该为6，后两个以此类推。

在本文中，则将用户在一段时间内购买的商品按顺序排列为一个session，用户下月购买的商品作为label，将所有用户的全部购买情况整理成对应的(session_list, label_list)数据格式。

处理后的数据文件config.txt、test.txt、train.txt三个文件下载后即可上传到脚本任务中训练。

### 数据预处理


```python
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import RobustScaler, LabelEncoder
tmp = pd.read_csv('data/data19383/train.csv')
```


```python
tmp = tmp[tmp.order_pay_time > '2013-02-01'][tmp.order_pay_time < '2013-09-01']
tmp['goods_id_ori'] = tmp['goods_id']
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.



```python
# 对商品id进行编码
tmp['goods_id'] = LabelEncoder().fit_transform(tmp['goods_id'])
```


```python
df_1 = tmp[['customer_id', 'goods_id', 'order_pay_time']][tmp.order_pay_time > '2013-01-01'][tmp.order_pay_time < '2013-04-01']
df_1 = df_1.sort_values(['customer_id', 'order_pay_time'])
df_1 = df_1['goods_id'].groupby(df_1['customer_id']).aggregate(lambda x:list(x)).reset_index()
label = tmp[['customer_id', 'goods_id']][tmp.order_pay_time > '2013-04-01'][tmp.order_pay_time < '2013-05-01']
label.rename(columns={'customer_id':'customer_id','goods_id':'label'}, inplace=True)
df_1 = pd.merge(df_1,label,how='left')
df_1.dropna(0,inplace=True)
df_1['label'] = df_1['label'].astype("int")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      after removing the cwd from sys.path.



```python
df_1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>goods_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1000737</td>
      <td>[45, 187, 295, 51, 86, 105, 0, 105]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1000737</td>
      <td>[45, 187, 295, 51, 86, 105, 0, 105]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_2 = tmp[['customer_id', 'goods_id', 'order_pay_time']][tmp.order_pay_time > '2013-02-01'][tmp.order_pay_time < '2013-05-01']
df_2 = df_2.sort_values(['customer_id', 'order_pay_time'])
df_2 = df_2['goods_id'].groupby(df_2['customer_id']).aggregate(lambda x:list(x)).reset_index()
label = tmp[['customer_id', 'goods_id']][tmp.order_pay_time > '2013-05-01'][tmp.order_pay_time < '2013-06-01']
label.rename(columns={'customer_id':'customer_id','goods_id':'label'}, inplace=True)
df_2 = pd.merge(df_2,label,how='left')
df_2.dropna(0,inplace=True)
df_2['label'] = df_2['label'].astype("int")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      after removing the cwd from sys.path.



```python
df_3 = tmp[['customer_id', 'goods_id', 'order_pay_time']][tmp.order_pay_time > '2013-03-01'][tmp.order_pay_time < '2013-06-01']
df_3 = df_3.sort_values(['customer_id', 'order_pay_time'])
df_3 = df_3['goods_id'].groupby(df_3['customer_id']).aggregate(lambda x:list(x)).reset_index()
label = tmp[['customer_id', 'goods_id']][tmp.order_pay_time > '2013-06-01'][tmp.order_pay_time < '2013-07-01']
label.rename(columns={'customer_id':'customer_id','goods_id':'label'}, inplace=True)
df_3 = pd.merge(df_3,label,how='left')
df_3.dropna(0,inplace=True)
df_3['label'] = df_3['label'].astype("int")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      after removing the cwd from sys.path.



```python
df_4 = tmp[['customer_id', 'goods_id', 'order_pay_time']][tmp.order_pay_time > '2013-04-01'][tmp.order_pay_time < '2013-07-01']
df_4 = df_4.sort_values(['customer_id', 'order_pay_time'])
df_4 = df_4['goods_id'].groupby(df_4['customer_id']).aggregate(lambda x:list(x)).reset_index()
label = tmp[['customer_id', 'goods_id']][tmp.order_pay_time > '2013-07-01'][tmp.order_pay_time < '2013-08-01']
label.rename(columns={'customer_id':'customer_id','goods_id':'label'}, inplace=True)
df_4 = pd.merge(df_4,label,how='left')
df_4.dropna(0,inplace=True)
df_4['label'] = df_4['label'].astype("int")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      after removing the cwd from sys.path.



```python
df_5 = tmp[['customer_id', 'goods_id', 'order_pay_time']][tmp.order_pay_time > '2013-05-01'][tmp.order_pay_time < '2013-08-01']
df_5 = df_5.sort_values(['customer_id', 'order_pay_time'])
df_5 = df_5['goods_id'].groupby(df_5['customer_id']).aggregate(lambda x:list(x)).reset_index()
label = tmp[['customer_id', 'goods_id']][tmp.order_pay_time > '2013-08-01'][tmp.order_pay_time < '2013-09-01']
label.rename(columns={'customer_id':'customer_id','goods_id':'label'}, inplace=True)
df_5 = pd.merge(df_5,label,how='left')
df_5.dropna(0,inplace=True)
df_5['label'] = df_5['label'].astype("int")
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:1: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      """Entry point for launching an IPython kernel.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/ipykernel_launcher.py:4: UserWarning: Boolean Series key will be reindexed to match DataFrame index.
      after removing the cwd from sys.path.



```python
df = df_1.append([df_2,df_3,df_4])
```


```python
# 生成的用户购买序列和标签
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>customer_id</th>
      <th>goods_id</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>12</th>
      <td>1000737</td>
      <td>[45, 187, 295, 51, 86, 105, 0, 105]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1000737</td>
      <td>[45, 187, 295, 51, 86, 105, 0, 105]</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1001232</td>
      <td>[156, 156, 156, 156, 156, 156, 156, 156]</td>
      <td>156</td>
    </tr>
  </tbody>
</table>
</div>




```python
df = df[['goods_id', 'label']]
df_5 = df_5[['goods_id', 'label']]
```


```python
tra = (list(df.goods_id), list(df.label))
tes = (list(df_5.goods_id), list(df_5.label))
```


```python
pickle.dump(tra, open('gnn/data/martech/train.txt', 'wb'))
```


```python
pickle.dump(tes, open('gnn/data/martech/test.txt', 'wb'))
```


```python
goods_seq = tmp[['goods_id','order_count']][tmp.order_pay_time > '2013-01-01'][tmp.order_pay_time < '2013-09-01']
```


```python
with open("gnn/data/martech/config.txt", "w") as fout:
    fout.write(str(len(goods_seq['goods_id'].unique())) + "\n")
```

### 训练和验证


```python
#使用GPU训练
!cd gnn && python train.py --train_path './data/martech/train.txt' --config_path './data/martech/config.txt' --use_cuda 1 --epoch_num 20 --model_path './saved_model'
```


```python
!cd gnn && python eval.py --model_path './saved_model' --test_path './data/martech/test.txt' --use_cuda 1
```

    2020-09-17 13:37:10,041-WARNING: paddle.fluid.layers.create_py_reader_by_data() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
    W0917 13:37:10.067281 11818 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
    W0917 13:37:10.071609 11818 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    2020-09-17 13:37:15,392-INFO: TEST -->Recall@20: 0.2432



```python
#预测,此处仅展示一例，按升序显示下月用户购买可能性最高的前三商品
!cd gnn && python infer.py --model_path './saved_model/' --test_path './data/martech/test.txt' --config_path './data/martech/config.txt' --use_cuda 1
```

    2020-09-17 13:37:56,643-WARNING: paddle.fluid.layers.create_py_reader_by_data() may be deprecated in the near future. Please use paddle.fluid.io.DataLoader.from_generator() instead.
    W0917 13:37:56.669899 12030 device_context.cc:252] Please NOTE: device: 0, CUDA Capability: 70, Driver API Version: 9.2, Runtime API Version: 9.0
    W0917 13:37:56.674283 12030 device_context.cc:260] device: 0, cuDNN Version: 7.6.
    pred: [134 437 156]



```python
# 打印正确的标签
pickle.load(open('gnn/data/martech/test.txt', 'rb'))[1][14]
```




    156


