# STGCN：路网交通预测
## 论文资料
- [Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic](https://arxiv.org/abs/1709.04875v4)
- [原文代码](https://github.com/VeritasYin/STGCN_IJCAI-18)
## 参考资料
- [论文阅读笔记](https://davidham3.github.io/blog/2018/05/10/spatio-temporal-graph-convolutional-networks-a-deep-learning-framework-for-traffic/)
- [mxnet复现代码](https://github.com/Davidham3/STGCN)
- [STGCN论文详解](https://zhuanlan.zhihu.com/p/78259670)
## 简介
实时精确的交通预测对城市交通管控和引导很重要。由于交通流的强非线性以及复杂性，传统方法并不能满足中长期预测的要求，而且传统方法经常忽略对时空数据的依赖。在这篇论文中，作者提出了一个新的深度学习框架，时空图卷积(Spatio-Temporal Graph Convolutional Networks)，来解决交通领域的时间序列预测问题。

在交通研究中，交通流的基本变量，也就是速度、流量和密度，通常作为监控当前交通状态以及未来预测的指示指标。根据预测的长度，交通预测大体分为两个尺度：短期(5~30min)，中和长期预测(超过30min)。大多数流行的统计方法(比如，线性回归)可以在短期预测上表现的很好。然而，由于交通流的不确定性和复杂性，这些方法在相对长期的预测上不是那么的有效。

交通预测是一个典型的时间序列预测问题，也就是预测在给定前M个观测样本接下来H个时间戳后最可能的交通流指标（比如速度或交通流）：

$\hat{v}_{t+1}, ..., \hat{v}_{t+H} = \mathop{\arg\min}_{v_{t+1},...,v_{t+H}}logP(v_{t+1},...,v_{t+H}\vert v_{t-M+1},...v_t)$

这里$v_t \in \mathbb{R}^n$是$n$个路段在时间戳$t$观察到的一个向量，每个元素记录了一条路段的历史观测数据。

作者在一个图上定义了一个交通网络，并专注于结构化的交通时间序列。观测到的样本v_t间不是相互独立的，而是在图中两两相互连接的。因此，数据点$v_t$可以被视为定义在权重为$w_{ij}$，如下图展示的无向图（或有向图）$\mathcal{G}$上的一个信号。在第$t$个时间戳，在图$\mathcal{G_t}=(\mathcal{V_t}, \mathcal{\varepsilon}, W), \mathcal{V_t}$是当顶点的有限集，对应在交通网络中n个监测站；$\epsilon$是边集，表示观测站之间的连通性；$W \in \mathbb{R^{n \times n}}$表示$\mathcal{G_t}$的邻接矩阵。

![file](https://pic1.zhimg.com/v2-5d5efa123e08b8e7cca00b327843aeb7_1440w.jpg)

> **关于图学习的概念，可以参考[PGL：Paddle带你走进图学习](https://aistudio.baidu.com/aistudio/projectdetail/413386)系列课程。**

> 关于交通网络的时空相关性，也可以参考其它论文给出的更详细示意，如：
>
> [论文：面向交通流量预测的多组件时空图卷积网络](http://www.jos.org.cn/html/2019/3/5697.htm)
>
> 交通流量预测是典型的时空数据预测问题, 不同类别的交通数据内嵌于连续空间, 并且随时间动态变化, 因此, 有效提取时空相关性对解决这类问题至关重要.下图所示为流量数据(也可以是车速、车道占用率等其他交通数据)的时空相关性示意图, 时间维包含3个时间片, 空间维的6个节点(A~F)表示公路的网状结构。在空间维上, 节点的交通状况之间会相互影响(绿色虚线); 时间维上, 某节点历史不同时刻流量会对该节点未来不同时刻流量产生影响(蓝色虚线); 同时, 节点历史不同时刻的流量值也会对其关联节点未来不同时刻的流量产生影响(红色虚线)。可见, 交通流量在时空维度都存在很强的相关性。
>
> ![file](http://www.jos.org.cn/html/2019/3/PIC/rjxb-30-3-759-1.jpg)

## 网络结构
STGCN有多个时空卷积块组成，每一个都是像一个“三明治”结构的组成，有两个门序列卷积层和一个空间图卷积层在中间。

![file](https://pic3.zhimg.com/80/v2-030389b5592ad95cc19e3546ae70510e_720w.jpg)

STGCN的架构有两个时空卷积块和一个全连接的在末尾的输出层组成。每个ST-Conv块包含了两个时间门卷积层，中间有一个空间图卷积层。每个块中都使用了残差连接和bottleneck策略。输入$v_{t-M+1},…v_t$被ST-Conv块均匀（uniformly）处理，来获取时空依赖关系。全部特征由一个输出层来整合，生成最后的预测$\hat{v}$。

## PGL实现
### 安装工具库


```python
!git clone https://gitee.com/paddlepaddle/PGL.git
```

    Cloning into 'PGL'...
    remote: Enumerating objects: 1824, done.[K
    remote: Counting objects: 100% (1824/1824), done.[K
    remote: Compressing objects: 100% (1224/1224), done.[K
    remote: Total 1824 (delta 977), reused 1061 (delta 528), pack-reused 0[K
    Receiving objects: 100% (1824/1824), 16.62 MiB | 3.40 MiB/s, done.
    Resolving deltas: 100% (977/977), done.
    Checking connectivity... done.



```python
!pip install pgl
```

    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Collecting pgl
    [?25l  Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/3f/d9/3a9db4a342545b1270cedf0ef68685108b1cf8cd2143a6aa5ee13ec2febf/pgl-1.1.0-cp37-cp37m-manylinux1_x86_64.whl (7.9MB)
    [K     |████████████████████████████████| 7.9MB 48kB/s eta 0:00:013
    [?25hCollecting redis-py-cluster (from pgl)
      Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/35/cb/29d44f7735af4fe9251afb6b5b173ec79c6e8f49cb6a61603e77a54ba658/redis_py_cluster-2.0.0-py2.py3-none-any.whl
    Requirement already satisfied: cython>=0.25.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (0.29)
    Requirement already satisfied: numpy>=1.16.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pgl) (1.16.4)
    Collecting redis<3.1.0,>=3.0.0 (from redis-py-cluster->pgl)
    [?25l  Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/f5/00/5253aff5e747faf10d8ceb35fb5569b848cde2fdc13685d42fcf63118bbc/redis-3.0.1-py2.py3-none-any.whl (61kB)
    [K     |████████████████████████████████| 71kB 32.9MB/s eta 0:00:01
    [?25hInstalling collected packages: redis, redis-py-cluster, pgl
    Successfully installed pgl-1.1.0 redis-3.0.1 redis-py-cluster-2.0.0


### 数据集准备
PeMSD7是Caltrans Performance Measurement System(PeMS)通过超过39000个监测站实时获取的数据，这些监测站分布在加州高速公路系统主要的都市部分。数据是30秒的数据样本聚合成5分钟一次的数据。作者在加州的District 7随机选取了一个小的和一个大的范围作为数据源，分别有228和1026个监测站，分别命名为PeMSD7(S)和PeMSD7(L)。PeMSD7数据集的时间范围是2012年五月和六月的周末，选取了第一个月的车速速度记录作为训练集，剩下的分别做验证和测试。

![file](https://ai-studio-static-online.cdn.bcebos.com/0ad23394ba10462993573f0c0e26014c80cdec467ded4176a4683e895bb54ac9)


### 数据预处理
路网中的每个顶点（传感器）每天就有288个数据点。数据清理后使用了线性插值的方法来填补缺失值。通过核对相关性，每条路的方向和OD(origin-destination)点，环路系统可以被数值化成一个有向图。

在PeMSD7，路网的邻接矩阵通过交通网络中的监测站的距离来计算。带权邻接矩阵W通过以下公式计算：

$w_{ij} = \begin{cases}
\exp{(-\frac{d^2_{ij}}{\sigma^2})}&,i \neq j \ \rm and \exp{(-\frac{d^2_{ij}}{\sigma^2}) \geq \epsilon} \\
0&, \rm otherwise
\end{cases}$

其中$w_{ij}$是边的权重，通过$d_{ij}$得到，也就是$i$和$j$之间的距离。$\sigma^2$和$\epsilon$是来控制矩阵W的分布和稀疏性的阈值，文中用了10和0.5。$W$的可视化在上图的右侧。

[作者代码](https://github.com/VeritasYin/STGCN_IJCAI-18)中给出了PeMS-M数据集的压缩包，本文已将数据集解压放在PeMS-M目录下。
```
PeMS-M/
    -- W_228.csv
    -- V_228.csv
```
其中，`V_228.csv`是传感器记录的车速信息，`W_228.csv`是已经处理好的邻接矩阵。

### 开始训练
> 由于没能跑通PGL给出的示例代码，这里对`PGL/examples/stgcn`目录下的`data_loader/data_utils.py`和`main.py`稍作修改，参考[mxnet复现代码](https://github.com/Davidham3/STGCN)重写了`data_loader/data_utils.py`里的部分函数。


```python
%cd PGL/examples/stgcn/
```

    /home/aistudio/PGL/examples/stgcn



```python
%run main.py --use_cuda --input_file /home/aistudio/PeMS-M/V_228.csv --adj_mat_file /home/aistudio/PeMS-M/W_228.csv
```

    [INFO] 2020-06-27 10:25:01,538 [     main.py:  174]:	Namespace(Ks=3, Kt=3, adj_mat_file='/home/aistudio/PeMS-M/W_228.csv', batch_size=10, blocks=[[1, 32, 64], [64, 32, 128]], epochs=5, inf_mode='sep', input_file='/home/aistudio/PeMS-M/V_228.csv', keep_prob=1.0, lr=0.001, n_his=9, n_pred=3, n_route=228, opt='ADAM', output_path='./outputs/', save=1, use_cuda=True)
    [INFO] 2020-06-27 10:25:04,711 [     main.py:   41]:	{'mean': 59.49499979949002, 'std': 13.170890048189376}
    [INFO] 2020-06-27 10:25:04,712 [     main.py:   42]:	7583
    [INFO] 2020-06-27 10:25:08,762 [     main.py:  109]:	epoch 1 | step 0 | lr 0.001000 | loss 37735.941406
    [INFO] 2020-06-27 10:25:13,379 [     main.py:  109]:	epoch 1 | step 5 | lr 0.001000 | loss 18736.484375
    [INFO] 2020-06-27 10:25:17,919 [     main.py:  109]:	epoch 1 | step 10 | lr 0.001000 | loss 18630.222656
    [INFO] 2020-06-27 10:25:22,470 [     main.py:  109]:	epoch 1 | step 15 | lr 0.001000 | loss 13020.451172
    [INFO] 2020-06-27 10:25:27,010 [     main.py:  109]:	epoch 1 | step 20 | lr 0.001000 | loss 18053.085938
    [INFO] 2020-06-27 10:25:31,652 [     main.py:  109]:	epoch 1 | step 25 | lr 0.001000 | loss 11902.833008
    [INFO] 2020-06-27 10:25:36,186 [     main.py:  109]:	epoch 1 | step 30 | lr 0.001000 | loss 11125.682617
    [INFO] 2020-06-27 10:25:40,800 [     main.py:  109]:	epoch 1 | step 35 | lr 0.001000 | loss 9599.474609
    [INFO] 2020-06-27 10:25:45,343 [     main.py:  109]:	epoch 1 | step 40 | lr 0.001000 | loss 11258.325195
    [INFO] 2020-06-27 10:25:49,907 [     main.py:  109]:	epoch 1 | step 45 | lr 0.001000 | loss 6970.545898
    [INFO] 2020-06-27 10:25:54,454 [     main.py:  109]:	epoch 1 | step 50 | lr 0.001000 | loss 7194.029785
    [INFO] 2020-06-27 10:25:58,999 [     main.py:  109]:	epoch 1 | step 55 | lr 0.001000 | loss 9328.666016
    [INFO] 2020-06-27 10:26:03,594 [     main.py:  109]:	epoch 1 | step 60 | lr 0.001000 | loss 5624.453613
    [INFO] 2020-06-27 10:26:08,158 [     main.py:  109]:	epoch 1 | step 65 | lr 0.001000 | loss 15484.669922
    [INFO] 2020-06-27 10:26:12,727 [     main.py:  109]:	epoch 1 | step 70 | lr 0.001000 | loss 9598.880859
    [INFO] 2020-06-27 10:26:17,282 [     main.py:  109]:	epoch 1 | step 75 | lr 0.001000 | loss 14367.545898
    [INFO] 2020-06-27 10:26:21,838 [     main.py:  109]:	epoch 1 | step 80 | lr 0.001000 | loss 9098.638672
    [INFO] 2020-06-27 10:26:26,381 [     main.py:  109]:	epoch 1 | step 85 | lr 0.001000 | loss 11955.281250
    [INFO] 2020-06-27 10:26:30,940 [     main.py:  109]:	epoch 1 | step 90 | lr 0.001000 | loss 7362.460938
    [INFO] 2020-06-27 10:26:35,509 [     main.py:  109]:	epoch 1 | step 95 | lr 0.001000 | loss 5939.260254
    [INFO] 2020-06-27 10:26:40,064 [     main.py:  109]:	epoch 1 | step 100 | lr 0.001000 | loss 16125.000977
    [INFO] 2020-06-27 10:26:44,755 [     main.py:  109]:	epoch 1 | step 105 | lr 0.001000 | loss 10326.945312
    [INFO] 2020-06-27 10:26:49,317 [     main.py:  109]:	epoch 1 | step 110 | lr 0.001000 | loss 16809.191406
    [INFO] 2020-06-27 10:26:53,874 [     main.py:  109]:	epoch 1 | step 115 | lr 0.001000 | loss 8291.509766
    [INFO] 2020-06-27 10:26:58,435 [     main.py:  109]:	epoch 1 | step 120 | lr 0.001000 | loss 8192.952148
    [INFO] 2020-06-27 10:27:03,006 [     main.py:  109]:	epoch 1 | step 125 | lr 0.001000 | loss 16092.328125
    [INFO] 2020-06-27 10:27:07,579 [     main.py:  109]:	epoch 1 | step 130 | lr 0.001000 | loss 6669.453125
    [INFO] 2020-06-27 10:27:12,164 [     main.py:  109]:	epoch 1 | step 135 | lr 0.001000 | loss 12109.406250
    [INFO] 2020-06-27 10:27:16,739 [     main.py:  109]:	epoch 1 | step 140 | lr 0.001000 | loss 6615.083008
    [INFO] 2020-06-27 10:27:21,313 [     main.py:  109]:	epoch 1 | step 145 | lr 0.001000 | loss 10833.454102
    [INFO] 2020-06-27 10:27:25,886 [     main.py:  109]:	epoch 1 | step 150 | lr 0.001000 | loss 9129.137695
    [INFO] 2020-06-27 10:27:30,512 [     main.py:  109]:	epoch 1 | step 155 | lr 0.001000 | loss 10883.227539
    [INFO] 2020-06-27 10:27:35,085 [     main.py:  109]:	epoch 1 | step 160 | lr 0.001000 | loss 6452.153320
    [INFO] 2020-06-27 10:27:39,677 [     main.py:  109]:	epoch 1 | step 165 | lr 0.001000 | loss 7721.391113
    [INFO] 2020-06-27 10:27:44,251 [     main.py:  109]:	epoch 1 | step 170 | lr 0.001000 | loss 6851.547852
    [INFO] 2020-06-27 10:27:48,833 [     main.py:  109]:	epoch 1 | step 175 | lr 0.001000 | loss 6517.803223
    [INFO] 2020-06-27 10:27:53,413 [     main.py:  109]:	epoch 1 | step 180 | lr 0.001000 | loss 5980.505859
    [INFO] 2020-06-27 10:27:57,990 [     main.py:  109]:	epoch 1 | step 185 | lr 0.001000 | loss 9012.660156
    [INFO] 2020-06-27 10:28:02,709 [     main.py:  109]:	epoch 1 | step 190 | lr 0.001000 | loss 8939.193359
    [INFO] 2020-06-27 10:28:07,284 [     main.py:  109]:	epoch 1 | step 195 | lr 0.001000 | loss 6330.277832
    [INFO] 2020-06-27 10:28:11,885 [     main.py:  109]:	epoch 1 | step 200 | lr 0.001000 | loss 6860.510742
    [INFO] 2020-06-27 10:28:16,480 [     main.py:  109]:	epoch 1 | step 205 | lr 0.001000 | loss 7429.390137
    [INFO] 2020-06-27 10:28:21,176 [     main.py:  109]:	epoch 1 | step 210 | lr 0.001000 | loss 5395.461426
    [INFO] 2020-06-27 10:28:25,846 [     main.py:  109]:	epoch 1 | step 215 | lr 0.001000 | loss 7995.499023
    [INFO] 2020-06-27 10:28:30,504 [     main.py:  109]:	epoch 1 | step 220 | lr 0.001000 | loss 10387.447266
    [INFO] 2020-06-27 10:28:35,176 [     main.py:  109]:	epoch 1 | step 225 | lr 0.001000 | loss 13723.822266
    [INFO] 2020-06-27 10:28:39,844 [     main.py:  109]:	epoch 1 | step 230 | lr 0.001000 | loss 6553.451172
    [INFO] 2020-06-27 10:28:44,502 [     main.py:  109]:	epoch 1 | step 235 | lr 0.001000 | loss 5084.278320
    [INFO] 2020-06-27 10:28:49,158 [     main.py:  109]:	epoch 1 | step 240 | lr 0.001000 | loss 11612.527344
    [INFO] 2020-06-27 10:28:53,817 [     main.py:  109]:	epoch 1 | step 245 | lr 0.001000 | loss 9042.700195
    [INFO] 2020-06-27 10:28:58,483 [     main.py:  109]:	epoch 1 | step 250 | lr 0.001000 | loss 5726.862793
    [INFO] 2020-06-27 10:29:03,150 [     main.py:  109]:	epoch 1 | step 255 | lr 0.001000 | loss 8533.449219
    [INFO] 2020-06-27 10:29:07,811 [     main.py:  109]:	epoch 1 | step 260 | lr 0.001000 | loss 6600.126953
    [INFO] 2020-06-27 10:29:12,647 [     main.py:  109]:	epoch 1 | step 265 | lr 0.001000 | loss 10771.875977
    [INFO] 2020-06-27 10:29:17,364 [     main.py:  109]:	epoch 1 | step 270 | lr 0.001000 | loss 7744.235840
    [INFO] 2020-06-27 10:29:22,020 [     main.py:  109]:	epoch 1 | step 275 | lr 0.001000 | loss 10922.890625
    [INFO] 2020-06-27 10:29:26,669 [     main.py:  109]:	epoch 1 | step 280 | lr 0.001000 | loss 10097.871094
    [INFO] 2020-06-27 10:29:31,338 [     main.py:  109]:	epoch 1 | step 285 | lr 0.001000 | loss 4943.081055
    [INFO] 2020-06-27 10:29:35,995 [     main.py:  109]:	epoch 1 | step 290 | lr 0.001000 | loss 5377.001953
    [INFO] 2020-06-27 10:29:40,655 [     main.py:  109]:	epoch 1 | step 295 | lr 0.001000 | loss 7947.578613
    [INFO] 2020-06-27 10:29:45,320 [     main.py:  109]:	epoch 1 | step 300 | lr 0.001000 | loss 10941.259766
    [INFO] 2020-06-27 10:29:49,972 [     main.py:  109]:	epoch 1 | step 305 | lr 0.001000 | loss 4228.699707
    [INFO] 2020-06-27 10:29:54,641 [     main.py:  109]:	epoch 1 | step 310 | lr 0.001000 | loss 6041.813965
    [INFO] 2020-06-27 10:29:59,298 [     main.py:  109]:	epoch 1 | step 315 | lr 0.001000 | loss 7565.720703
    [INFO] 2020-06-27 10:30:03,955 [     main.py:  109]:	epoch 1 | step 320 | lr 0.001000 | loss 3052.132812
    [INFO] 2020-06-27 10:30:08,664 [     main.py:  109]:	epoch 1 | step 325 | lr 0.001000 | loss 7258.500000
    [INFO] 2020-06-27 10:30:13,318 [     main.py:  109]:	epoch 1 | step 330 | lr 0.001000 | loss 7258.628906
    [INFO] 2020-06-27 10:30:17,971 [     main.py:  109]:	epoch 1 | step 335 | lr 0.001000 | loss 10175.652344
    [INFO] 2020-06-27 10:30:22,633 [     main.py:  109]:	epoch 1 | step 340 | lr 0.001000 | loss 6401.521973
    [INFO] 2020-06-27 10:30:27,445 [     main.py:  109]:	epoch 1 | step 345 | lr 0.001000 | loss 3520.176270
    [INFO] 2020-06-27 10:30:32,091 [     main.py:  109]:	epoch 1 | step 350 | lr 0.001000 | loss 5676.759766
    [INFO] 2020-06-27 10:30:36,752 [     main.py:  109]:	epoch 1 | step 355 | lr 0.001000 | loss 7543.050781
    [INFO] 2020-06-27 10:30:41,407 [     main.py:  109]:	epoch 1 | step 360 | lr 0.001000 | loss 12806.579102
    [INFO] 2020-06-27 10:30:46,057 [     main.py:  109]:	epoch 1 | step 365 | lr 0.001000 | loss 4636.592285
    [INFO] 2020-06-27 10:30:50,719 [     main.py:  109]:	epoch 1 | step 370 | lr 0.001000 | loss 6429.077637
    [INFO] 2020-06-27 10:30:55,372 [     main.py:  109]:	epoch 1 | step 375 | lr 0.001000 | loss 6336.032715
    [INFO] 2020-06-27 10:31:00,031 [     main.py:  109]:	epoch 1 | step 380 | lr 0.001000 | loss 8332.381836
    [INFO] 2020-06-27 10:31:04,703 [     main.py:  109]:	epoch 1 | step 385 | lr 0.001000 | loss 6266.290039
    [INFO] 2020-06-27 10:31:09,363 [     main.py:  109]:	epoch 1 | step 390 | lr 0.001000 | loss 6936.368164
    [INFO] 2020-06-27 10:31:14,036 [     main.py:  109]:	epoch 1 | step 395 | lr 0.001000 | loss 4875.728027
    [INFO] 2020-06-27 10:31:18,712 [     main.py:  109]:	epoch 1 | step 400 | lr 0.001000 | loss 9043.921875
    [INFO] 2020-06-27 10:31:23,378 [     main.py:  109]:	epoch 1 | step 405 | lr 0.001000 | loss 6416.957520
    [INFO] 2020-06-27 10:31:28,052 [     main.py:  109]:	epoch 1 | step 410 | lr 0.001000 | loss 6685.189453
    [INFO] 2020-06-27 10:31:32,723 [     main.py:  109]:	epoch 1 | step 415 | lr 0.001000 | loss 8260.904297
    [INFO] 2020-06-27 10:31:37,545 [     main.py:  109]:	epoch 1 | step 420 | lr 0.001000 | loss 5355.673828
    [INFO] 2020-06-27 10:31:42,203 [     main.py:  109]:	epoch 1 | step 425 | lr 0.001000 | loss 3357.874023
    [INFO] 2020-06-27 10:31:46,868 [     main.py:  109]:	epoch 1 | step 430 | lr 0.001000 | loss 6597.340820
    [INFO] 2020-06-27 10:31:51,537 [     main.py:  109]:	epoch 1 | step 435 | lr 0.001000 | loss 5140.948242
    [INFO] 2020-06-27 10:31:56,203 [     main.py:  109]:	epoch 1 | step 440 | lr 0.001000 | loss 14872.176758
    [INFO] 2020-06-27 10:32:00,870 [     main.py:  109]:	epoch 1 | step 445 | lr 0.001000 | loss 5137.071777
    [INFO] 2020-06-27 10:32:05,536 [     main.py:  109]:	epoch 1 | step 450 | lr 0.001000 | loss 3784.291504
    [INFO] 2020-06-27 10:32:10,207 [     main.py:  109]:	epoch 1 | step 455 | lr 0.001000 | loss 5173.709473
    [INFO] 2020-06-27 10:32:14,881 [     main.py:  109]:	epoch 1 | step 460 | lr 0.001000 | loss 5030.171387
    [INFO] 2020-06-27 10:32:19,553 [     main.py:  109]:	epoch 1 | step 465 | lr 0.001000 | loss 5143.199707
    [INFO] 2020-06-27 10:32:24,226 [     main.py:  109]:	epoch 1 | step 470 | lr 0.001000 | loss 9018.249023
    [INFO] 2020-06-27 10:32:28,893 [     main.py:  109]:	epoch 1 | step 475 | lr 0.001000 | loss 4279.703613
    [INFO] 2020-06-27 10:32:33,558 [     main.py:  109]:	epoch 1 | step 480 | lr 0.001000 | loss 7475.649902
    [INFO] 2020-06-27 10:32:38,226 [     main.py:  109]:	epoch 1 | step 485 | lr 0.001000 | loss 8123.766113
    [INFO] 2020-06-27 10:32:42,905 [     main.py:  109]:	epoch 1 | step 490 | lr 0.001000 | loss 7915.218750
    [INFO] 2020-06-27 10:32:47,742 [     main.py:  109]:	epoch 1 | step 495 | lr 0.001000 | loss 5127.339844
    [INFO] 2020-06-27 10:32:52,519 [     main.py:  109]:	epoch 1 | step 500 | lr 0.001000 | loss 4054.055908
    [INFO] 2020-06-27 10:32:57,171 [     main.py:  109]:	epoch 1 | step 505 | lr 0.001000 | loss 4345.636719
    [INFO] 2020-06-27 10:33:01,835 [     main.py:  109]:	epoch 1 | step 510 | lr 0.001000 | loss 7016.473633
    [INFO] 2020-06-27 10:33:06,489 [     main.py:  109]:	epoch 1 | step 515 | lr 0.001000 | loss 7494.670410
    [INFO] 2020-06-27 10:33:11,164 [     main.py:  109]:	epoch 1 | step 520 | lr 0.001000 | loss 6326.111328
    [INFO] 2020-06-27 10:33:15,824 [     main.py:  109]:	epoch 1 | step 525 | lr 0.001000 | loss 5116.449219
    [INFO] 2020-06-27 10:33:20,487 [     main.py:  109]:	epoch 1 | step 530 | lr 0.001000 | loss 5296.669922
    [INFO] 2020-06-27 10:33:25,145 [     main.py:  109]:	epoch 1 | step 535 | lr 0.001000 | loss 6458.634766
    [INFO] 2020-06-27 10:33:29,802 [     main.py:  109]:	epoch 1 | step 540 | lr 0.001000 | loss 5442.626953
    [INFO] 2020-06-27 10:33:34,467 [     main.py:  109]:	epoch 1 | step 545 | lr 0.001000 | loss 6636.049316
    [INFO] 2020-06-27 10:33:39,140 [     main.py:  109]:	epoch 1 | step 550 | lr 0.001000 | loss 8034.599121
    [INFO] 2020-06-27 10:33:43,804 [     main.py:  109]:	epoch 1 | step 555 | lr 0.001000 | loss 5106.713867
    [INFO] 2020-06-27 10:33:48,467 [     main.py:  109]:	epoch 1 | step 560 | lr 0.001000 | loss 7601.837891
    [INFO] 2020-06-27 10:33:53,135 [     main.py:  109]:	epoch 1 | step 565 | lr 0.001000 | loss 5841.292969
    [INFO] 2020-06-27 10:33:57,791 [     main.py:  109]:	epoch 1 | step 570 | lr 0.001000 | loss 2632.594727
    [INFO] 2020-06-27 10:34:02,614 [     main.py:  109]:	epoch 1 | step 575 | lr 0.001000 | loss 5246.858398
    [INFO] 2020-06-27 10:34:07,272 [     main.py:  109]:	epoch 1 | step 580 | lr 0.001000 | loss 6611.925293
    [INFO] 2020-06-27 10:34:11,931 [     main.py:  109]:	epoch 1 | step 585 | lr 0.001000 | loss 6576.196289
    [INFO] 2020-06-27 10:34:16,589 [     main.py:  109]:	epoch 1 | step 590 | lr 0.001000 | loss 7234.600586
    [INFO] 2020-06-27 10:34:21,246 [     main.py:  109]:	epoch 1 | step 595 | lr 0.001000 | loss 7896.226074
    [INFO] 2020-06-27 10:34:25,895 [     main.py:  109]:	epoch 1 | step 600 | lr 0.001000 | loss 7527.335449
    [INFO] 2020-06-27 10:34:30,553 [     main.py:  109]:	epoch 1 | step 605 | lr 0.001000 | loss 3311.047363
    [INFO] 2020-06-27 10:34:35,223 [     main.py:  109]:	epoch 1 | step 610 | lr 0.001000 | loss 3356.222168
    [INFO] 2020-06-27 10:34:39,879 [     main.py:  109]:	epoch 1 | step 615 | lr 0.001000 | loss 6924.463379
    [INFO] 2020-06-27 10:34:44,537 [     main.py:  109]:	epoch 1 | step 620 | lr 0.001000 | loss 4124.032715
    [INFO] 2020-06-27 10:34:49,202 [     main.py:  109]:	epoch 1 | step 625 | lr 0.001000 | loss 5382.447754
    [INFO] 2020-06-27 10:34:53,861 [     main.py:  109]:	epoch 1 | step 630 | lr 0.001000 | loss 5890.448242
    [INFO] 2020-06-27 10:34:58,557 [     main.py:  109]:	epoch 1 | step 635 | lr 0.001000 | loss 4065.146484
    [INFO] 2020-06-27 10:35:03,221 [     main.py:  109]:	epoch 1 | step 640 | lr 0.001000 | loss 4500.604980
    [INFO] 2020-06-27 10:35:07,893 [     main.py:  109]:	epoch 1 | step 645 | lr 0.001000 | loss 4620.673828
    [INFO] 2020-06-27 10:35:12,724 [     main.py:  109]:	epoch 1 | step 650 | lr 0.001000 | loss 4947.170898
    [INFO] 2020-06-27 10:35:17,379 [     main.py:  109]:	epoch 1 | step 655 | lr 0.001000 | loss 6553.805664
    [INFO] 2020-06-27 10:35:22,033 [     main.py:  109]:	epoch 1 | step 660 | lr 0.001000 | loss 6870.678223
    [INFO] 2020-06-27 10:35:26,678 [     main.py:  109]:	epoch 1 | step 665 | lr 0.001000 | loss 6609.209961
    [INFO] 2020-06-27 10:35:31,345 [     main.py:  109]:	epoch 1 | step 670 | lr 0.001000 | loss 5503.839355
    [INFO] 2020-06-27 10:35:36,000 [     main.py:  109]:	epoch 1 | step 675 | lr 0.001000 | loss 4718.570312
    [INFO] 2020-06-27 10:35:40,654 [     main.py:  109]:	epoch 1 | step 680 | lr 0.001000 | loss 5936.292480
    [INFO] 2020-06-27 10:35:45,322 [     main.py:  109]:	epoch 1 | step 685 | lr 0.001000 | loss 4619.918945
    [INFO] 2020-06-27 10:35:49,972 [     main.py:  109]:	epoch 1 | step 690 | lr 0.001000 | loss 5239.690430
    [INFO] 2020-06-27 10:35:54,642 [     main.py:  109]:	epoch 1 | step 695 | lr 0.001000 | loss 7379.635742
    [INFO] 2020-06-27 10:35:59,294 [     main.py:  109]:	epoch 1 | step 700 | lr 0.001000 | loss 3559.312500
    [INFO] 2020-06-27 10:36:03,952 [     main.py:  109]:	epoch 1 | step 705 | lr 0.001000 | loss 4938.158691
    [INFO] 2020-06-27 10:36:08,614 [     main.py:  109]:	epoch 1 | step 710 | lr 0.001000 | loss 6022.327637
    [INFO] 2020-06-27 10:36:13,333 [     main.py:  109]:	epoch 1 | step 715 | lr 0.001000 | loss 3227.090088
    [INFO] 2020-06-27 10:36:17,996 [     main.py:  109]:	epoch 1 | step 720 | lr 0.001000 | loss 4335.575195
    [INFO] 2020-06-27 10:36:22,672 [     main.py:  109]:	epoch 1 | step 725 | lr 0.001000 | loss 7386.066895
    [INFO] 2020-06-27 10:36:27,489 [     main.py:  109]:	epoch 1 | step 730 | lr 0.001000 | loss 4961.151367
    [INFO] 2020-06-27 10:36:32,147 [     main.py:  109]:	epoch 1 | step 735 | lr 0.001000 | loss 3725.366699
    [INFO] 2020-06-27 10:36:36,805 [     main.py:  109]:	epoch 1 | step 740 | lr 0.001000 | loss 5296.148926
    [INFO] 2020-06-27 10:36:41,466 [     main.py:  109]:	epoch 1 | step 745 | lr 0.001000 | loss 3624.866943
    [INFO] 2020-06-27 10:36:46,128 [     main.py:  109]:	epoch 1 | step 750 | lr 0.001000 | loss 3798.115234
    [INFO] 2020-06-27 10:36:50,798 [     main.py:  109]:	epoch 1 | step 755 | lr 0.001000 | loss 4630.307129


    Time Step 3: MAPE 13.896%, 12.535%; MAE  4.914, 4.759; RMSE  7.973,  7.636.
    Time Step 1: MAPE 11.567%; MAE  4.552; RMSE  7.401.
    Time Step 2: MAPE 11.927%; MAE  4.595; RMSE  7.437.
    Time Step 3: MAPE 12.535%; MAE  4.759; RMSE  7.636.


    [INFO] 2020-06-27 10:44:26,944 [     main.py:  109]:	epoch 2 | step 0 | lr 0.001000 | loss 5626.972656
    [INFO] 2020-06-27 10:44:31,626 [     main.py:  109]:	epoch 2 | step 5 | lr 0.001000 | loss 5328.616699
    [INFO] 2020-06-27 10:44:36,273 [     main.py:  109]:	epoch 2 | step 10 | lr 0.001000 | loss 4990.459961
    [INFO] 2020-06-27 10:44:40,917 [     main.py:  109]:	epoch 2 | step 15 | lr 0.001000 | loss 4208.468750
    [INFO] 2020-06-27 10:44:45,592 [     main.py:  109]:	epoch 2 | step 20 | lr 0.001000 | loss 5790.794922
    [INFO] 2020-06-27 10:44:50,268 [     main.py:  109]:	epoch 2 | step 25 | lr 0.001000 | loss 4302.013672
    [INFO] 2020-06-27 10:44:54,961 [     main.py:  109]:	epoch 2 | step 30 | lr 0.001000 | loss 6110.498535
    [INFO] 2020-06-27 10:44:59,642 [     main.py:  109]:	epoch 2 | step 35 | lr 0.001000 | loss 6567.183594
    [INFO] 2020-06-27 10:45:04,327 [     main.py:  109]:	epoch 2 | step 40 | lr 0.001000 | loss 3542.793457
    [INFO] 2020-06-27 10:45:08,983 [     main.py:  109]:	epoch 2 | step 45 | lr 0.001000 | loss 4186.571777


# STGCN：高致病性传染病传播趋势预测基线系统学习
- [比赛链接](https://aistudio.baidu.com/aistudio/competition/detail/36)
- [基于飞桨PGL的基线系统](https://aistudio.baidu.com/aistudio/projectdetail/464528)

在该场景中，官方baseline使用STGCN进行传播趋势预测，并没有改动图结构和STGCN网络，重点更多在数据预处理上。与交通流量的自回归不同的是，增加了label也就是感染人数。

本文基于官方baseline略作修改，并增加了数据预处理阶段的注释，目录结构如下：
```
work/
  -- dataset/
  -- dataloader.py 
  -- Dataset.py
  -- graph.py
  -- model.py
  -- main.py
```

## 解压数据集


```python
%cd /home/aistudio/work
```

    /home/aistudio/work



```python
# unzip dataset
%mkdir ./dataset/
!unzip /home/aistudio/data/data33637/train_data.zip -d ./dataset/
```

    Archive:  /home/aistudio/data/data33637/train_data.zip
       creating: ./dataset/train_data/
       creating: ./dataset/train_data/city_C/
      inflating: ./dataset/train_data/city_C/migration.csv  
      inflating: ./dataset/train_data/city_C/density.csv  
      inflating: ./dataset/train_data/city_C/transfer.csv  
      inflating: ./dataset/train_data/city_C/weather.csv  
      inflating: ./dataset/train_data/city_C/grid_attr.csv  
      inflating: ./dataset/train_data/city_C/infection.csv  
       creating: ./dataset/train_data/city_D/
      inflating: ./dataset/train_data/city_D/density.csv  
      inflating: ./dataset/train_data/city_D/migration.csv  
      inflating: ./dataset/train_data/city_D/transfer.csv  
      inflating: ./dataset/train_data/city_D/weather.csv  
      inflating: ./dataset/train_data/city_D/grid_attr.csv  
      inflating: ./dataset/train_data/city_D/infection.csv  
       creating: ./dataset/train_data/city_E/
      inflating: ./dataset/train_data/city_E/density.csv  
      inflating: ./dataset/train_data/city_E/migration.csv  
      inflating: ./dataset/train_data/city_E/transfer.csv  
      inflating: ./dataset/train_data/city_E/weather.csv  
      inflating: ./dataset/train_data/city_E/grid_attr.csv  
      inflating: ./dataset/train_data/city_E/infection.csv  
      inflating: ./dataset/train_data/submission.csv  
       creating: ./dataset/train_data/city_A/
      inflating: ./dataset/train_data/city_A/density.csv  
      inflating: ./dataset/train_data/city_A/migration.csv  
      inflating: ./dataset/train_data/city_A/transfer.csv  
      inflating: ./dataset/train_data/city_A/weather.csv  
      inflating: ./dataset/train_data/city_A/grid_attr.csv  
      inflating: ./dataset/train_data/city_A/infection.csv  
       creating: ./dataset/train_data/city_B/
      inflating: ./dataset/train_data/city_B/density.csv  
      inflating: ./dataset/train_data/city_B/migration.csv  
      inflating: ./dataset/train_data/city_B/transfer.csv  
      inflating: ./dataset/train_data/city_B/weather.csv  
      inflating: ./dataset/train_data/city_B/grid_attr.csv  
      inflating: ./dataset/train_data/city_B/infection.csv  


## 引入工具库


```python
"""data processing
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict, OrderedDict
import pdb
import gc
```

## 获取指定城市每个地点经纬度以及归属区域
根据`grid_attr.csv`给出的城市地点经纬度详细信息，将其转为`<key,value>`的形式。

其实这种做法有个前提，那就是每个城市的`grid_attr.csv`包括了迁移数据中起始点和到达点的全部经纬度。假设在真实场景下，可以考虑用电子围栏。


```python
def get_grid_dict(city_path, city_name):
    d = {}
    with open(os.path.join(city_path, 'grid_attr.csv'), 'r') as f:
        for line in f:
            items = line.strip().split(',')
            axis = ",".join(items[0:2])
            ID = items[2]
            d[axis] = "_".join([city_name, ID])
    # print(d)
    # d = {'x,y': ID}
    return d
```

## 计算市内区域迁移指数
反映人群迁移情况的`transfer.csv`里起始点和到达点都是经纬度，这里需要将其换算为所属的城市区域ID，通过查找`grid_dict`里面的`<key,value>`列表实现。


```python
def coord2ID(data_path, city_name, output_path):
    city_path = os.path.join(data_path, "city_%s" % city_name)
    grid_dict = get_grid_dict(city_path, city_name)
    # grid_dict = {'x,y': ID}
    trans_filename = os.path.join(city_path, "transfer.csv")
    output_file = os.path.join(output_path, "%s_transfer.csv" % (city_name))
    with open(trans_filename, 'r') as f, open(output_file, 'w') as writer:
        for line in f:
            items = line.strip().split(',')
            start_axis = ",".join(items[1:3])
            end_axis = ",".join(items[3:5])
            index = items[5]
            try:
                start_ID = grid_dict[start_axis]
                end_ID = grid_dict[end_axis] 
            except KeyError: # remove no ID axis
                continue

            writer.write("%s,%s,%s,%s\n" % (items[0], start_ID, end_ID, index))
```


```python
coord2ID('./dataset/train_data', 'A', './dataset')
```


```python
# 查看得到的每小时区域人群迁移指数
pd.read_csv('dataset/A_transfer.csv', header=None, names=['hour', 's_region', 'e_region', 'index']).head()
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
      <th>hour</th>
      <th>s_region</th>
      <th>e_region</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>0.1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 求和计算每日市内区域人群迁移指数
def calc_index_in_one_day(data_path, city_name):
    trans_filename = os.path.join(data_path, "%s_transfer.csv" % (city_name))
    transfer = pd.read_csv(trans_filename, 
            header=None,
            names=['hour', 's_region', 'e_region', 'index'])
        
    df = transfer.groupby(['s_region', 'e_region'])['index'].sum().reset_index()
    df = df[['s_region', 'e_region', 'index']]
    #  df = df.T
    #  df_list.append(df)
    return df
```


```python
calc_index_in_one_day('./dataset', 'A')
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
      <th>s_region</th>
      <th>e_region</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A_0</td>
      <td>A_0</td>
      <td>187.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A_0</td>
      <td>A_1</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A_0</td>
      <td>A_10</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A_0</td>
      <td>A_100</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A_0</td>
      <td>A_108</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>A_0</td>
      <td>A_11</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>A_0</td>
      <td>A_110</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>A_0</td>
      <td>A_111</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>A_0</td>
      <td>A_112</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>9</th>
      <td>A_0</td>
      <td>A_116</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>A_0</td>
      <td>A_117</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>A_0</td>
      <td>A_12</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>A_0</td>
      <td>A_13</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>13</th>
      <td>A_0</td>
      <td>A_14</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>14</th>
      <td>A_0</td>
      <td>A_15</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>15</th>
      <td>A_0</td>
      <td>A_16</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>16</th>
      <td>A_0</td>
      <td>A_17</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>A_0</td>
      <td>A_18</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>18</th>
      <td>A_0</td>
      <td>A_19</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>19</th>
      <td>A_0</td>
      <td>A_2</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>20</th>
      <td>A_0</td>
      <td>A_20</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>21</th>
      <td>A_0</td>
      <td>A_23</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>22</th>
      <td>A_0</td>
      <td>A_25</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>A_0</td>
      <td>A_26</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>A_0</td>
      <td>A_29</td>
      <td>2.7</td>
    </tr>
    <tr>
      <th>25</th>
      <td>A_0</td>
      <td>A_30</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>26</th>
      <td>A_0</td>
      <td>A_31</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>27</th>
      <td>A_0</td>
      <td>A_32</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>A_0</td>
      <td>A_33</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>A_0</td>
      <td>A_34</td>
      <td>0.2</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>13026</th>
      <td>A_99</td>
      <td>A_72</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>13027</th>
      <td>A_99</td>
      <td>A_73</td>
      <td>0.7</td>
    </tr>
    <tr>
      <th>13028</th>
      <td>A_99</td>
      <td>A_74</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>13029</th>
      <td>A_99</td>
      <td>A_75</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>13030</th>
      <td>A_99</td>
      <td>A_76</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>13031</th>
      <td>A_99</td>
      <td>A_77</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>13032</th>
      <td>A_99</td>
      <td>A_78</td>
      <td>5.4</td>
    </tr>
    <tr>
      <th>13033</th>
      <td>A_99</td>
      <td>A_79</td>
      <td>6.6</td>
    </tr>
    <tr>
      <th>13034</th>
      <td>A_99</td>
      <td>A_8</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>13035</th>
      <td>A_99</td>
      <td>A_80</td>
      <td>5.7</td>
    </tr>
    <tr>
      <th>13036</th>
      <td>A_99</td>
      <td>A_81</td>
      <td>9.3</td>
    </tr>
    <tr>
      <th>13037</th>
      <td>A_99</td>
      <td>A_82</td>
      <td>3.8</td>
    </tr>
    <tr>
      <th>13038</th>
      <td>A_99</td>
      <td>A_83</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>13039</th>
      <td>A_99</td>
      <td>A_84</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>13040</th>
      <td>A_99</td>
      <td>A_85</td>
      <td>0.4</td>
    </tr>
    <tr>
      <th>13041</th>
      <td>A_99</td>
      <td>A_86</td>
      <td>0.1</td>
    </tr>
    <tr>
      <th>13042</th>
      <td>A_99</td>
      <td>A_87</td>
      <td>26.8</td>
    </tr>
    <tr>
      <th>13043</th>
      <td>A_99</td>
      <td>A_88</td>
      <td>32.7</td>
    </tr>
    <tr>
      <th>13044</th>
      <td>A_99</td>
      <td>A_89</td>
      <td>7.6</td>
    </tr>
    <tr>
      <th>13045</th>
      <td>A_99</td>
      <td>A_9</td>
      <td>1.4</td>
    </tr>
    <tr>
      <th>13046</th>
      <td>A_99</td>
      <td>A_90</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>13047</th>
      <td>A_99</td>
      <td>A_91</td>
      <td>0.3</td>
    </tr>
    <tr>
      <th>13048</th>
      <td>A_99</td>
      <td>A_92</td>
      <td>69.0</td>
    </tr>
    <tr>
      <th>13049</th>
      <td>A_99</td>
      <td>A_93</td>
      <td>119.5</td>
    </tr>
    <tr>
      <th>13050</th>
      <td>A_99</td>
      <td>A_94</td>
      <td>50.9</td>
    </tr>
    <tr>
      <th>13051</th>
      <td>A_99</td>
      <td>A_95</td>
      <td>3.9</td>
    </tr>
    <tr>
      <th>13052</th>
      <td>A_99</td>
      <td>A_96</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>13053</th>
      <td>A_99</td>
      <td>A_97</td>
      <td>1.2</td>
    </tr>
    <tr>
      <th>13054</th>
      <td>A_99</td>
      <td>A_98</td>
      <td>135.4</td>
    </tr>
    <tr>
      <th>13055</th>
      <td>A_99</td>
      <td>A_99</td>
      <td>153.0</td>
    </tr>
  </tbody>
</table>
<p>13056 rows × 3 columns</p>
</div>



## 计算城市迁移指数
每个城市的`migration.csv`记录了从该城市出发、到达该城市的人流量指数，可以通过统计计算每日到达指定城市的人流量。


```python
def process_city_migration(data_path, city_name):
    filename = os.path.join(data_path, "city_%s" % city_name, "migration.csv")
    migration = pd.read_csv(filename, 
                            sep=',', 
                            header=None,
                            names=['date', 's_city', 'e_city', city_name])

    # only use moving in "city" data, ignore moving out data
    df = migration[migration.e_city == city_name]
    df = df[["date", city_name]]

    # calculate total move in data of "city"
    df = df.groupby('date')[city_name].sum().reset_index()
    return df
```


```python
# 计算每日到达A地的人流量
process_city_migration('./dataset/train_data', 'A')
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
      <th>date</th>
      <th>A</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21200501</td>
      <td>0.811620</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21200502</td>
      <td>0.742641</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21200503</td>
      <td>0.964937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21200504</td>
      <td>0.771767</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21200505</td>
      <td>0.727024</td>
    </tr>
    <tr>
      <th>5</th>
      <td>21200506</td>
      <td>1.101211</td>
    </tr>
    <tr>
      <th>6</th>
      <td>21200507</td>
      <td>0.750903</td>
    </tr>
    <tr>
      <th>7</th>
      <td>21200508</td>
      <td>1.004562</td>
    </tr>
    <tr>
      <th>8</th>
      <td>21200509</td>
      <td>0.887760</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21200510</td>
      <td>0.890514</td>
    </tr>
    <tr>
      <th>10</th>
      <td>21200511</td>
      <td>1.074288</td>
    </tr>
    <tr>
      <th>11</th>
      <td>21200512</td>
      <td>0.802903</td>
    </tr>
    <tr>
      <th>12</th>
      <td>21200513</td>
      <td>0.715456</td>
    </tr>
    <tr>
      <th>13</th>
      <td>21200514</td>
      <td>0.713415</td>
    </tr>
    <tr>
      <th>14</th>
      <td>21200515</td>
      <td>0.724205</td>
    </tr>
    <tr>
      <th>15</th>
      <td>21200516</td>
      <td>0.846061</td>
    </tr>
    <tr>
      <th>16</th>
      <td>21200517</td>
      <td>0.784372</td>
    </tr>
    <tr>
      <th>17</th>
      <td>21200518</td>
      <td>1.017717</td>
    </tr>
    <tr>
      <th>18</th>
      <td>21200519</td>
      <td>0.842400</td>
    </tr>
    <tr>
      <th>19</th>
      <td>21200520</td>
      <td>0.751421</td>
    </tr>
    <tr>
      <th>20</th>
      <td>21200521</td>
      <td>1.040493</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21200522</td>
      <td>0.796489</td>
    </tr>
    <tr>
      <th>22</th>
      <td>21200523</td>
      <td>1.011236</td>
    </tr>
    <tr>
      <th>23</th>
      <td>21200524</td>
      <td>0.708232</td>
    </tr>
    <tr>
      <th>24</th>
      <td>21200525</td>
      <td>0.730554</td>
    </tr>
    <tr>
      <th>25</th>
      <td>21200526</td>
      <td>0.740178</td>
    </tr>
    <tr>
      <th>26</th>
      <td>21200527</td>
      <td>0.714582</td>
    </tr>
    <tr>
      <th>27</th>
      <td>21200528</td>
      <td>0.935291</td>
    </tr>
    <tr>
      <th>28</th>
      <td>21200529</td>
      <td>0.759197</td>
    </tr>
    <tr>
      <th>29</th>
      <td>21200530</td>
      <td>1.102637</td>
    </tr>
    <tr>
      <th>30</th>
      <td>21200531</td>
      <td>0.955054</td>
    </tr>
    <tr>
      <th>31</th>
      <td>21200601</td>
      <td>0.733666</td>
    </tr>
    <tr>
      <th>32</th>
      <td>21200602</td>
      <td>0.669124</td>
    </tr>
    <tr>
      <th>33</th>
      <td>21200603</td>
      <td>0.669902</td>
    </tr>
    <tr>
      <th>34</th>
      <td>21200604</td>
      <td>0.684385</td>
    </tr>
    <tr>
      <th>35</th>
      <td>21200605</td>
      <td>0.541469</td>
    </tr>
    <tr>
      <th>36</th>
      <td>21200606</td>
      <td>0.664135</td>
    </tr>
    <tr>
      <th>37</th>
      <td>21200607</td>
      <td>0.669902</td>
    </tr>
    <tr>
      <th>38</th>
      <td>21200608</td>
      <td>0.508906</td>
    </tr>
    <tr>
      <th>39</th>
      <td>21200609</td>
      <td>0.139514</td>
    </tr>
    <tr>
      <th>40</th>
      <td>21200610</td>
      <td>0.684385</td>
    </tr>
    <tr>
      <th>41</th>
      <td>21200611</td>
      <td>0.379826</td>
    </tr>
    <tr>
      <th>42</th>
      <td>21200612</td>
      <td>0.770699</td>
    </tr>
    <tr>
      <th>43</th>
      <td>21200613</td>
      <td>0.508906</td>
    </tr>
    <tr>
      <th>44</th>
      <td>21200614</td>
      <td>0.541469</td>
    </tr>
  </tbody>
</table>
</div>




```python
def migration_process(data_path, city_list, output_path):
    for city_name in city_list:
        coord2ID(data_path, city_name, output_path)
        transfer = calc_index_in_one_day(output_path, city_name)
        migration = process_city_migration(data_path, city_name)

        df_list = []
        for i in range(len(migration)):
            df = transfer.copy()
            date = migration.date[i]
            index = migration[city_name][i]
            # 这里通过将到达城市的人流量指数与市内区域人流量指数相乘，得到区域人流量指数
            df['index'] = df['index'] * index
            df['date'] = date
            df = df[['date', 's_region', 'e_region', 'index']]
            # 按日新增区域人流量统计数据
            df_list.append(df)

        df = pd.concat(df_list, axis=0)
        # 得到每个城市最终迁移人流量指数
        df.to_csv(os.path.join(output_path, '%s_migration.csv' % city_name), 
                header=None,
                index=None,
                float_format = '%.4f')
```


```python
migration_process('./dataset/train_data', ["A", "B", "C", "D", "E"], './dataset')
```


```python
pd.read_csv('dataset/A_migration.csv', header=None).head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21200501</td>
      <td>A_0</td>
      <td>A_0</td>
      <td>152.1787</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21200501</td>
      <td>A_0</td>
      <td>A_1</td>
      <td>0.2435</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21200501</td>
      <td>A_0</td>
      <td>A_10</td>
      <td>0.6493</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21200501</td>
      <td>A_0</td>
      <td>A_100</td>
      <td>0.1623</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21200501</td>
      <td>A_0</td>
      <td>A_108</td>
      <td>0.3246</td>
    </tr>
  </tbody>
</table>
</div>



## 计算邻接矩阵


```python
def adj_matrix_process(data_path, city_list, region_nums, output_path):
    total_region_num = np.sum(region_nums)
    adj_matrix = np.zeros((total_region_num, total_region_num))

    offset = 0
    for i, city in enumerate(city_list):
        filename = os.path.join(output_path, "%s_migration.csv" % city)
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=['date', 's_region', 'e_region', 'index'])
        # 生成每个城市的初始邻接矩阵，比如A城市是有118个区域，shape就是(118, 118)
        matrix = np.zeros((region_nums[i], region_nums[i]))
        # pdb.set_trace()
        # 对区域编码进行排序，官方baseline的写法会把10,100排到2,3,4等前面
        # order = sorted(range(region_nums[i]), key=lambda x:str(x))
        order = sorted(list(range(region_nums[i])))
        for j, idx in enumerate(order):
            # 拼接目标区域的标准名称：城市名称+区域ID
            target_region = "%s_%d" % (city, idx)
            # only use moving in "city" data, ignore moving out data
            # 只用到迁入城市的人流量，不考虑迁出的问题
            df = migration[migration['e_region'] == target_region]

            # 计算得到每个区域迁入的平均人流量
            df = df.groupby('s_region')['index'].mean().reset_index()
            #  res = df['index'].values.reshape(-1)
            for k, o in enumerate(order):
                s_region_id = "%s_%d" % (city, o)
                try:
                    # 取出来自指定出发地的平均人流量数据
                    value = df[df['s_region'] == s_region_id]['index'].values[0]
                except:
                    value = 0.0
                if s_region_id == target_region:
                    value = 0.0
                # 给邻接矩阵该位置的元素赋值
                matrix[j, k] = value

        # merge two adj_matrix
        # 把不同城市的邻接矩阵拼起来，形成最终的大的邻接矩阵
        adj_matrix[offset:(offset + region_nums[i]), offset:(offset + region_nums[i])] = matrix
        offset += region_nums[i]
    # 这里调整了一下，保存为csv格式
    file_to_save = os.path.join(output_path, 'adj_matrix.csv')
    print("saving result to %s" % file_to_save)
    # np.save(file_to_save, adj_matrix)
    np.savetxt(file_to_save, adj_matrix, delimiter=',')
```


```python
adj_matrix_process('./dataset/train_data', ["A", "B", "C", "D", "E"], [118, 30, 135, 75, 34], './dataset')
```

    saving result to ./dataset/adj_matrix.csv



```python
pd.read_csv('dataset/adj_matrix.csv',header=None).head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>382</th>
      <th>383</th>
      <th>384</th>
      <th>385</th>
      <th>386</th>
      <th>387</th>
      <th>388</th>
      <th>389</th>
      <th>390</th>
      <th>391</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.000000</td>
      <td>0.076853</td>
      <td>0.307436</td>
      <td>0.000000</td>
      <td>0.384284</td>
      <td>0.000000</td>
      <td>0.230571</td>
      <td>0.230571</td>
      <td>0.000000</td>
      <td>0.230571</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.230571</td>
      <td>0.000000</td>
      <td>0.153711</td>
      <td>0.230571</td>
      <td>0.076853</td>
      <td>0.999138</td>
      <td>0.076853</td>
      <td>0.153711</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.153711</td>
      <td>0.307436</td>
      <td>0.000000</td>
      <td>3.996533</td>
      <td>13.219304</td>
      <td>0.537987</td>
      <td>0.845416</td>
      <td>1.767704</td>
      <td>1.460271</td>
      <td>4.611380</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.535398</td>
      <td>0.000000</td>
      <td>8.377353</td>
      <td>0.076853</td>
      <td>6.993933</td>
      <td>1.844553</td>
      <td>1.152851</td>
      <td>5.379951</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.230571</td>
      <td>0.000000</td>
      <td>14.372151</td>
      <td>8.069929</td>
      <td>0.000000</td>
      <td>0.922280</td>
      <td>2.613129</td>
      <td>7.685642</td>
      <td>4.841956</td>
      <td>18.906689</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 392 columns</p>
</div>



## 感染人数处理


```python
pd.read_csv('dataset/train_data/city_A/infection.csv', header=None, names=["city", "region", "date", "infect"]).head()
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
      <th>city</th>
      <th>region</th>
      <th>date</th>
      <th>infect</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>A</td>
      <td>0</td>
      <td>21200501</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A</td>
      <td>0</td>
      <td>21200502</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>A</td>
      <td>0</td>
      <td>21200503</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>A</td>
      <td>0</td>
      <td>21200504</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>A</td>
      <td>0</td>
      <td>21200505</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
def infection_process(data_path, city_list, region_nums, output_path):
    res = []
    region_name_list = []
    for i, city in enumerate(city_list):
        filename = os.path.join(data_path, "city_%s" % city, "infection.csv")
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=["city", "region", "date", "infect"])

        # order = sorted(range(region_nums[i]), key=lambda x:str(x))
        order = sorted(list(range(region_nums[i])))
        for j, idx in enumerate(order):
            target_region = idx #str(idx)
            # pdb.set_trace()
            # 区域每天感染人数
            df = migration[migration['region'] == target_region].reset_index(drop=True)
            if i == 0 and j == 0:
                # 第一个区域要把日期给进去
                df = df[['date', 'infect']]
            else:
                df = df[['infect']]

            df = df.rename(columns={'infect': '%s_%d' % (city, idx)})
            region_name_list.append("%s_%d" % (city, idx))

            res.append(df)
    df = pd.concat(res, axis=1)
    # 最终形成城市+区域ID形式的感染人数大宽表
    file_to_save = os.path.join(output_path, "infection.csv")
    print("saving result to %s" % file_to_save)
    # format: [date, A, B, C, D, E]
    df.to_csv(file_to_save, index=False)

    region_name_file = os.path.join(output_path, "region_names.txt")
    with open(region_name_file, 'w') as f:
        names = ' '.join(region_name_list)
        f.write(names + '\n')
```


```python
infection_process('./dataset/train_data', ["A", "B", "C", "D", "E"], [118, 30, 135, 75, 34], './dataset')
```

    saving result to ./dataset/infection.csv



```python
pd.read_csv('dataset/infection.csv').head()
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
      <th>date</th>
      <th>A_0</th>
      <th>A_1</th>
      <th>A_2</th>
      <th>A_3</th>
      <th>A_4</th>
      <th>A_5</th>
      <th>A_6</th>
      <th>A_7</th>
      <th>A_8</th>
      <th>...</th>
      <th>E_24</th>
      <th>E_25</th>
      <th>E_26</th>
      <th>E_27</th>
      <th>E_28</th>
      <th>E_29</th>
      <th>E_30</th>
      <th>E_31</th>
      <th>E_32</th>
      <th>E_33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21200501</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21200502</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21200503</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21200504</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21200505</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 393 columns</p>
</div>



## 迁移人流量处理


```python
def region_migration_process(data_path, city_list, region_nums, output_path):
    res = []
    # 这里和感染人数处理类似
    for i, city in enumerate(city_list):
        filename = os.path.join(output_path, "%s_migration.csv" % city)
        migration = pd.read_csv(filename, 
                                sep=',', 
                                header=None,
                                names=['date', 's_region', 'e_region', 'index'])

        # order = sorted(range(region_nums[i]), key=lambda x:str(x))
        order = sorted(list(range(region_nums[i])))
        for j, idx in enumerate(order):
            target_region = "%s_%d" % (city, idx)
            df = migration[migration['e_region'] == target_region]

            df = df.groupby('date')['index'].sum().reset_index()

            if i == 0 and j == 0:
                df = df[['date', 'index']]
            else:
                df = df[['index']]

            df = df.rename(columns={'index': target_region})

            res.append(df)
    # 最终形成城市+区域ID形式的迁移人流量大宽表
    df = pd.concat(res, axis=1)

    file_to_save = os.path.join(output_path, "region_migration.csv")
    print("saving result to %s" % file_to_save)
    # format: [date, A, B, C, D, E]
    df.to_csv(file_to_save, index=False, float_format = '%.2f')
```


```python
region_migration_process('./dataset/train_data', ["A", "B", "C", "D", "E"], [118, 30, 135, 75, 34], './dataset')
```

    saving result to ./dataset/region_migration.csv



```python
pd.read_csv('dataset/region_migration.csv').head()
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
      <th>date</th>
      <th>A_0</th>
      <th>A_1</th>
      <th>A_2</th>
      <th>A_3</th>
      <th>A_4</th>
      <th>A_5</th>
      <th>A_6</th>
      <th>A_7</th>
      <th>A_8</th>
      <th>...</th>
      <th>E_24</th>
      <th>E_25</th>
      <th>E_26</th>
      <th>E_27</th>
      <th>E_28</th>
      <th>E_29</th>
      <th>E_30</th>
      <th>E_31</th>
      <th>E_32</th>
      <th>E_33</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>21200501</td>
      <td>171.82</td>
      <td>200.06</td>
      <td>231.88</td>
      <td>209.15</td>
      <td>403.13</td>
      <td>179.94</td>
      <td>204.45</td>
      <td>897.98</td>
      <td>1156.07</td>
      <td>...</td>
      <td>63.98</td>
      <td>50.18</td>
      <td>50.74</td>
      <td>64.47</td>
      <td>21.42</td>
      <td>7.04</td>
      <td>11.47</td>
      <td>NaN</td>
      <td>9.93</td>
      <td>3.14</td>
    </tr>
    <tr>
      <th>1</th>
      <td>21200502</td>
      <td>157.22</td>
      <td>183.06</td>
      <td>212.17</td>
      <td>191.38</td>
      <td>368.87</td>
      <td>164.64</td>
      <td>187.07</td>
      <td>821.66</td>
      <td>1057.82</td>
      <td>...</td>
      <td>56.44</td>
      <td>44.27</td>
      <td>44.76</td>
      <td>56.87</td>
      <td>18.90</td>
      <td>6.21</td>
      <td>10.12</td>
      <td>NaN</td>
      <td>8.76</td>
      <td>2.77</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21200503</td>
      <td>204.28</td>
      <td>237.86</td>
      <td>275.68</td>
      <td>248.66</td>
      <td>479.28</td>
      <td>213.93</td>
      <td>243.07</td>
      <td>1067.61</td>
      <td>1374.46</td>
      <td>...</td>
      <td>70.30</td>
      <td>55.13</td>
      <td>55.75</td>
      <td>70.83</td>
      <td>23.54</td>
      <td>7.73</td>
      <td>12.60</td>
      <td>NaN</td>
      <td>10.91</td>
      <td>3.45</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21200504</td>
      <td>163.38</td>
      <td>190.24</td>
      <td>220.49</td>
      <td>198.89</td>
      <td>383.34</td>
      <td>171.10</td>
      <td>194.41</td>
      <td>853.88</td>
      <td>1099.30</td>
      <td>...</td>
      <td>58.88</td>
      <td>46.18</td>
      <td>46.69</td>
      <td>59.33</td>
      <td>19.72</td>
      <td>6.47</td>
      <td>10.55</td>
      <td>NaN</td>
      <td>9.14</td>
      <td>2.89</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21200505</td>
      <td>153.91</td>
      <td>179.21</td>
      <td>207.71</td>
      <td>187.35</td>
      <td>361.11</td>
      <td>161.18</td>
      <td>183.14</td>
      <td>804.38</td>
      <td>1035.57</td>
      <td>...</td>
      <td>57.64</td>
      <td>45.20</td>
      <td>45.71</td>
      <td>58.07</td>
      <td>19.30</td>
      <td>6.34</td>
      <td>10.33</td>
      <td>NaN</td>
      <td>8.95</td>
      <td>2.83</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 393 columns</p>
</div>




```python
%run main.py --batch_size 1
```

    [INFO] 2020-06-26 14:28:43,150 [     main.py:  219]:	Namespace(Ks=3, Kt=4, adj_mat_file='./dataset/adj_matrix.csv', batch_size=1, blocks=[[1, 16, 32], [32, 16, 64]], city_num=392, epochs=10, feat_dim=1, input_file='./dataset/region_migration.csv', keep_prob=1.0, label_file='./dataset/infection.csv', lr=0.005, n_his=20, n_pred=1, opt='ADAM', output_path='../outputs/', region_names_file='./dataset/region_names.txt', save=5, seed=1, submit_file='./dataset/train_data/submission.csv', test_num=1, use_cuda=False, val_num=3)
    [INFO] 2020-06-26 14:28:44,674 [     main.py:   39]:	num examples: 26
    [INFO] 2020-06-26 14:28:44,675 [     main.py:   45]:	Train examples: 22
    [INFO] 2020-06-26 14:28:44,676 [     main.py:   46]:	Test examples: 1
    [INFO] 2020-06-26 14:28:44,676 [     main.py:   50]:	Valid examples: 3


    region migration:         date      A_0      A_1      A_2      A_3      A_4      A_5      A_6  \
    0  21200501  0.17182  0.20006  0.23188  0.20915  0.40313  0.17994  0.20445   
    1  21200502  0.15722  0.18306  0.21217  0.19138  0.36887  0.16464  0.18707   
    2  21200503  0.20428  0.23786  0.27568  0.24866  0.47928  0.21393  0.24307   
    3  21200504  0.16338  0.19024  0.22049  0.19889  0.38334  0.17110  0.19441   
    4  21200505  0.15391  0.17921  0.20771  0.18735  0.36111  0.16118  0.18314   
    
           A_7      A_8   ...        E_24     E_25     E_26     E_27     E_28  \
    0  0.89798  1.15607   ...     0.06398  0.05018  0.05074  0.06447  0.02142   
    1  0.82166  1.05782   ...     0.05644  0.04427  0.04476  0.05687  0.01890   
    2  1.06761  1.37446   ...     0.07030  0.05513  0.05575  0.07083  0.02354   
    3  0.85388  1.09930   ...     0.05888  0.04618  0.04669  0.05933  0.01972   
    4  0.80438  1.03557   ...     0.05764  0.04520  0.04571  0.05807  0.01930   
    
          E_29     E_30  E_31     E_32     E_33  
    0  0.00704  0.01147   0.0  0.00993  0.00314  
    1  0.00621  0.01012   0.0  0.00876  0.00277  
    2  0.00773  0.01260   0.0  0.01091  0.00345  
    3  0.00647  0.01055   0.0  0.00914  0.00289  
    4  0.00634  0.01033   0.0  0.00895  0.00283  
    
    [5 rows x 393 columns]
    infect:         date  A_0  A_1  A_2  A_3  A_4  A_5  A_6  A_7  A_8  ...   E_24  E_25  \
    0  21200501  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0   0.0   
    1  21200502  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0   0.0   
    2  21200503  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0   0.0   
    3  21200504  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0   0.0   
    4  21200505  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  0.0  ...    0.0   0.0   
    
       E_26  E_27  E_28  E_29  E_30  E_31  E_32  E_33  
    0   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
    1   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
    2   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
    3   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
    4   0.0   0.0   0.0   0.0   0.0   0.0   0.0   0.0  
    
    [5 rows x 393 columns]


    [INFO] 2020-06-26 14:29:22,067 [     main.py:  103]:	epoch 1 | step 5 | loss 5141.372070
    [INFO] 2020-06-26 14:29:57,194 [     main.py:  103]:	epoch 1 | step 10 | loss 14939.403320
    [INFO] 2020-06-26 14:30:12,754 [     main.py:  114]:	valid result: | rmsle 0.3194172598309787 
    [INFO] 2020-06-26 14:32:13,501 [     main.py:  103]:	epoch 1 | step 15 | loss 45625.109375
    [INFO] 2020-06-26 14:32:48,288 [     main.py:  103]:	epoch 1 | step 20 | loss 2329.699219
    [INFO] 2020-06-26 14:33:03,864 [     main.py:  114]:	valid result: | rmsle 0.3262588392553634 
    [INFO] 2020-06-26 14:33:38,870 [     main.py:  103]:	epoch 2 | step 25 | loss 1218.109741
    [INFO] 2020-06-26 14:34:14,048 [     main.py:  103]:	epoch 2 | step 30 | loss 3981.177734
    [INFO] 2020-06-26 14:34:30,069 [     main.py:  114]:	valid result: | rmsle 0.31293732133227786 
    [INFO] 2020-06-26 14:36:28,008 [     main.py:  103]:	epoch 2 | step 35 | loss 51102.667969
    [INFO] 2020-06-26 14:37:02,589 [     main.py:  103]:	epoch 2 | step 40 | loss 4952.343262
    [INFO] 2020-06-26 14:37:18,472 [     main.py:  114]:	valid result: | rmsle 0.3501166642559801 
    [INFO] 2020-06-26 14:37:53,013 [     main.py:  103]:	epoch 3 | step 45 | loss 1652.163452
    [INFO] 2020-06-26 14:38:27,370 [     main.py:  103]:	epoch 3 | step 50 | loss 1507.490601
    [INFO] 2020-06-26 14:38:43,138 [     main.py:  114]:	valid result: | rmsle 0.331668068887688 
    [INFO] 2020-06-26 14:39:17,488 [     main.py:  103]:	epoch 3 | step 55 | loss 21587.957031
    [INFO] 2020-06-26 14:39:52,009 [     main.py:  103]:	epoch 3 | step 60 | loss 33287.515625
    [INFO] 2020-06-26 14:40:07,656 [     main.py:  114]:	valid result: | rmsle 0.44680587784508113 
    [INFO] 2020-06-26 14:40:41,938 [     main.py:  103]:	epoch 3 | step 65 | loss 1946.974976
    [INFO] 2020-06-26 14:41:15,824 [     main.py:  103]:	epoch 4 | step 70 | loss 1601.681396
    [INFO] 2020-06-26 14:41:31,381 [     main.py:  114]:	valid result: | rmsle 0.33738222738482104 
    [INFO] 2020-06-26 14:42:05,358 [     main.py:  103]:	epoch 4 | step 75 | loss 6963.198242
    [INFO] 2020-06-26 14:42:39,445 [     main.py:  103]:	epoch 4 | step 80 | loss 44315.679688
    [INFO] 2020-06-26 14:42:55,124 [     main.py:  114]:	valid result: | rmsle 0.4920501171182198 
    [INFO] 2020-06-26 14:43:29,205 [     main.py:  103]:	epoch 4 | step 85 | loss 2973.472900
    [INFO] 2020-06-26 14:44:03,215 [     main.py:  103]:	epoch 5 | step 90 | loss 2044.652954
    [INFO] 2020-06-26 14:44:18,849 [     main.py:  114]:	valid result: | rmsle 0.34906841902967184 
    [INFO] 2020-06-26 14:44:52,696 [     main.py:  103]:	epoch 5 | step 95 | loss 2296.614746
    [INFO] 2020-06-26 14:45:26,765 [     main.py:  103]:	epoch 5 | step 100 | loss 32906.230469
    [INFO] 2020-06-26 14:45:42,225 [     main.py:  114]:	valid result: | rmsle 0.3861702873697199 
    [INFO] 2020-06-26 14:46:16,294 [     main.py:  103]:	epoch 5 | step 105 | loss 16162.965820
    [INFO] 2020-06-26 14:46:50,673 [     main.py:  103]:	epoch 5 | step 110 | loss 2610.524170
    [INFO] 2020-06-26 14:47:06,688 [     main.py:  114]:	valid result: | rmsle 0.36127532195454476 
    [INFO] 2020-06-26 14:47:41,135 [     main.py:  103]:	epoch 6 | step 115 | loss 1780.250732
    [INFO] 2020-06-26 14:48:15,459 [     main.py:  103]:	epoch 6 | step 120 | loss 10502.518555
    [INFO] 2020-06-26 14:48:31,190 [     main.py:  114]:	valid result: | rmsle 0.3130958394801865 
    [INFO] 2020-06-26 14:49:05,392 [     main.py:  103]:	epoch 6 | step 125 | loss 35854.316406
    [INFO] 2020-06-26 14:49:39,823 [     main.py:  103]:	epoch 6 | step 130 | loss 2997.706787
    [INFO] 2020-06-26 14:49:55,792 [     main.py:  114]:	valid result: | rmsle 0.34820518475746004 
    [INFO] 2020-06-26 14:50:30,806 [     main.py:  103]:	epoch 7 | step 135 | loss 2552.869141
    [INFO] 2020-06-26 14:51:05,198 [     main.py:  103]:	epoch 7 | step 140 | loss 3003.099121
    [INFO] 2020-06-26 14:51:20,733 [     main.py:  114]:	valid result: | rmsle 0.3653511384072864 
    [INFO] 2020-06-26 14:51:54,912 [     main.py:  103]:	epoch 7 | step 145 | loss 40328.570312
    [INFO] 2020-06-26 14:52:29,192 [     main.py:  103]:	epoch 7 | step 150 | loss 3527.086670
    [INFO] 2020-06-26 14:52:45,366 [     main.py:  114]:	valid result: | rmsle 0.4565134441321422 
    [INFO] 2020-06-26 14:53:19,548 [     main.py:  103]:	epoch 8 | step 155 | loss 3548.696045
    [INFO] 2020-06-26 14:53:54,068 [     main.py:  103]:	epoch 8 | step 160 | loss 1838.059937
    [INFO] 2020-06-26 14:54:09,859 [     main.py:  114]:	valid result: | rmsle 0.3835494900254456 
    [INFO] 2020-06-26 14:54:43,921 [     main.py:  103]:	epoch 8 | step 165 | loss 16120.377930
    [INFO] 2020-06-26 14:55:18,154 [     main.py:  103]:	epoch 8 | step 170 | loss 26304.166016
    [INFO] 2020-06-26 14:55:34,055 [     main.py:  114]:	valid result: | rmsle 0.47084887527540514 
    [INFO] 2020-06-26 14:56:08,455 [     main.py:  103]:	epoch 8 | step 175 | loss 4118.551758
    [INFO] 2020-06-26 14:56:42,696 [     main.py:  103]:	epoch 9 | step 180 | loss 3040.967285
    [INFO] 2020-06-26 14:56:58,365 [     main.py:  114]:	valid result: | rmsle 0.37827040902171283 
    [INFO] 2020-06-26 14:57:32,668 [     main.py:  103]:	epoch 9 | step 185 | loss 4641.135742
    [INFO] 2020-06-26 14:58:06,828 [     main.py:  103]:	epoch 9 | step 190 | loss 36007.191406
    [INFO] 2020-06-26 14:58:22,481 [     main.py:  114]:	valid result: | rmsle 0.3609482683381391 
    [INFO] 2020-06-26 14:58:56,981 [     main.py:  103]:	epoch 9 | step 195 | loss 2561.531738
    [INFO] 2020-06-26 14:59:31,859 [     main.py:  103]:	epoch 10 | step 200 | loss 4813.833008
    [INFO] 2020-06-26 14:59:48,230 [     main.py:  114]:	valid result: | rmsle 0.3453318222476775 
    [INFO] 2020-06-26 15:00:23,296 [     main.py:  103]:	epoch 10 | step 205 | loss 1748.335938
    [INFO] 2020-06-26 15:00:58,044 [     main.py:  103]:	epoch 10 | step 210 | loss 27014.339844
    [INFO] 2020-06-26 15:01:13,904 [     main.py:  114]:	valid result: | rmsle 0.3741153056705634 
    [INFO] 2020-06-26 15:01:48,446 [     main.py:  103]:	epoch 10 | step 215 | loss 12417.771484
    [INFO] 2020-06-26 15:02:22,801 [     main.py:  103]:	epoch 10 | step 220 | loss 4376.041016
    [INFO] 2020-06-26 15:02:38,670 [     main.py:  114]:	valid result: | rmsle 0.394630465021254 
    [INFO] 2020-06-26 15:02:38,671 [     main.py:  122]:	best valid result: 0.31293732133227786


# 交通流量预测
## 数据集介绍
### 路段属性表
每条道路的每个通行方向由多条路段（link）构成，数据集中会提供每条link的唯一标识，长度，宽度，以及道路类型。

![file](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279213540/1530604123635_UwFWal5fag.jpg)


```python
gy_link_info = pd.read_csv('data/data40468/gy_link_info.txt',sep=';')
gy_link_info.head()
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
      <th>link_ID</th>
      <th>length</th>
      <th>width</th>
      <th>link_class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4377906289869500514</td>
      <td>57</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4377906284594800514</td>
      <td>247</td>
      <td>9</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4377906289425800514</td>
      <td>194</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4377906284525800514</td>
      <td>839</td>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4377906284422600514</td>
      <td>55</td>
      <td>12</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



###  link上下游关系表
link之间按照车辆允许通行的方向存在上下游关系，数据集中提供每条link的直接上游link和直接下游link。

![file](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279213540/1530604200365_YWdoFt080j.jpg)


```python
gy_link_top = pd.read_csv('data/data40468/gy_link_top.txt',sep=';')
gy_link_top.head()
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
      <th>link_ID</th>
      <th>in_links</th>
      <th>out_links</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4377906289869500514</td>
      <td>4377906285525800514</td>
      <td>4377906281969500514</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4377906284594800514</td>
      <td>4377906284514600514</td>
      <td>4377906285594800514</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4377906289425800514</td>
      <td>NaN</td>
      <td>4377906284653600514</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4377906284525800514</td>
      <td>4377906281234600514</td>
      <td>4377906280334600514</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4377906284422600514</td>
      <td>3377906289434510514#4377906287959500514</td>
      <td>4377906283422600514</td>
    </tr>
  </tbody>
</table>
</div>



### link历史通行时间表
数据集中记录了历史每天不同时间段内（2min为一个时间段）每条link上的平均旅行时间，每个时间段的平均旅行时间是基于在该时间段内进入link的车辆在该link上的旅行时间产出。

![file](http://aliyuntianchipublic.cn-hangzhou.oss-pub.aliyun-inc.com/public/files/image/1095279213540/1530604254989_39m8JehJpp.jpg)


```python
!unzip data/data40468/travel_time_1.zip -d work/dataset/
!unzip data/data40468/travel_time_2.zip -d work/dataset/
!unzip data/data40468/travel_time_3.zip -d work/dataset/
```

    Archive:  data/data40468/travel_time_1.zip
    replace work/dataset/gy_link_travel_time_part1.txt? [y]es, [n]o, [A]ll, [N]one, [r]ename: ^C
    Archive:  data/data40468/travel_time_2.zip
    replace work/dataset/gy_link_travel_time_part2.txt? [y]es, [n]o, [A]ll, [N]one, [r]ename: 


```python
df1 = pd.read_csv('work/dataset/gy_link_travel_time_part1.txt',sep=';')
df2 = pd.read_csv('work/dataset/gy_link_travel_time_part2.txt',sep=';', names=['link_ID','date','time_interval','travel_time'])
df2.drop(0,inplace=True)
df3 = pd.read_csv('work/dataset/gy_link_travel_time_part3.txt',sep=';')
```

    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/IPython/core/interactiveshell.py:3018: DtypeWarning: Columns (0,3) have mixed types. Specify dtype option on import or set low_memory=False.
      interactivity=interactivity, compiler=compiler, result=result)



```python
gy_link_travel_time = pd.concat([df1, df2, df3], axis=0)
gy_link_travel_time.tail()
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
      <th>link_ID</th>
      <th>date</th>
      <th>time_interval</th>
      <th>travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10632039</th>
      <td>4377906287663800514</td>
      <td>2017-07-31</td>
      <td>[2017-07-31 14:16:00,2017-07-31 14:18:00)</td>
      <td>76.9</td>
    </tr>
    <tr>
      <th>10632040</th>
      <td>4377906288663800514</td>
      <td>2017-07-31</td>
      <td>[2017-07-31 07:08:00,2017-07-31 07:10:00)</td>
      <td>3.4</td>
    </tr>
    <tr>
      <th>10632041</th>
      <td>4377906288663800514</td>
      <td>2017-07-31</td>
      <td>[2017-07-31 17:12:00,2017-07-31 17:14:00)</td>
      <td>5.1</td>
    </tr>
    <tr>
      <th>10632042</th>
      <td>4377906288663800514</td>
      <td>2017-07-31</td>
      <td>[2017-07-31 17:38:00,2017-07-31 17:40:00)</td>
      <td>36.6</td>
    </tr>
    <tr>
      <th>10632043</th>
      <td>3377906289044510514</td>
      <td>2017-07-31</td>
      <td>[2017-07-31 17:08:00,2017-07-31 17:10:00)</td>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>




```python
del df1
del df2
del df3
gc.collect()
```


```python
gy_link_travel_time['time_interval'] = gy_link_travel_time['time_interval'].str(1:20)
```


```python
len(gy_link_travel_time['link_ID'].unique())
```




    264




```python
tmp = gy_link_travel_time[['time_interval', 'link_ID', 'travel_time']]
```


```python
tmp.to_csv('work/dataset/gy_link_travel_time.csv',index=None)
```


```python
tmp = pd.read_csv('work/dataset/gy_link_travel_time.csv',parse_dates=['travel_time'])
```


```python
tmp.head()
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
      <th>time_interval</th>
      <th>link_ID</th>
      <th>travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-05-21 23:20:00</td>
      <td>9377906285566510514</td>
      <td>17.6</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-05-21 18:46:00</td>
      <td>3377906288228510514</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-05-21 07:06:00</td>
      <td>3377906284395510514</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-05-21 14:34:00</td>
      <td>4377906284959500514</td>
      <td>3.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-05-21 05:04:00</td>
      <td>9377906282776510514</td>
      <td>1.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
tmp['travel_time'] = tmp['travel_time'].astype(np.float16)
```


```python
tmp = tmp.groupby(['time_interval','link_ID']).agg({'travel_time': ['mean']})
tmp.columns = ['travel_time']
tmp.reset_index(inplace=True)
```


```python
tmp.head()
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
      <th>time_interval</th>
      <th>link_ID</th>
      <th>travel_time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-03-01 00:00:00</td>
      <td>3377906280028510514</td>
      <td>4.601562</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-03-01 00:00:00</td>
      <td>3377906280395510514</td>
      <td>22.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-03-01 00:00:00</td>
      <td>3377906282328510514</td>
      <td>20.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-03-01 00:00:00</td>
      <td>3377906283328510514</td>
      <td>6.601562</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-03-01 00:00:00</td>
      <td>3377906284028510514</td>
      <td>19.593750</td>
    </tr>
  </tbody>
</table>
</div>



### 生成完整时间段序列，解决空值问题


```python
tmp['time_interval'].unique()
```




    array(['2016-03-01 00:00:00', '2016-03-01 00:02:00',
           '2016-03-01 00:04:00', ..., '2017-07-31 17:54:00',
           '2017-07-31 17:56:00', '2017-07-31 17:58:00'], dtype=object)




```python
len(tmp['time_interval'].unique())
```




    183624




```python
# 生成完整的时间段序列
ts = pd.Series(np.zeros(len(tmp['time_interval'].unique())), index=tmp['time_interval'].unique())
ts.to_csv("data/ts.csv", header=None)
ts = pd.read_csv("data/ts.csv", names=['time_interval', 'value'])
```


```python
# tmp = pd.merge(ts,tmp,how='left')
# tmp.drop(['value'], axis=1, inplace=True)
```

## 依样画葫芦
### 通行时间处理


```python
def car_process(tmp, ts, output_path):
    res = []
    link_name_list = []
        # order = sorted(range(region_nums[i]), key=lambda x:str(x))
    order = sorted(tmp['link_ID'].unique())
    for i, idx in enumerate(order):
        target_link = idx #str(idx)
        # pdb.set_trace()
        # 路段平均车速
        df = tmp[tmp['link_ID'] == target_link].reset_index(drop=True)
        df = pd.merge(ts,df,how='left')
        df.drop(['value'], axis=1, inplace=True)
        if i == 0:
            # 第一个路段要把时间给进去
            df = df[['time_interval', 'travel_time']]
        else:
            df = df[['travel_time']]

        df = df.rename(columns={'travel_time': '%d' % (idx)})
        link_name_list.append("%d" % (idx))

        res.append(df)
    df = pd.concat(res, axis=1)
    # 最终形成路段ID形式的平均车速大宽表
    file_to_save = os.path.join(output_path, "travel_time.csv")
    print("saving result to %s" % file_to_save)
    df.to_csv(file_to_save, index=False)

    link_name_file = os.path.join(output_path, "link_name_list.txt")
    with open(link_name_file, 'w') as f:
        names = ' '.join(link_name_list)
        # print(names)
        f.write(names)
```


```python
car_process(tmp,ts,'work/dataset')
```

    saving result to work/dataset/travel_time.csv



```python
travel_time = pd.read_csv('work/dataset/travel_time.csv')
```


```python
travel_time.tail()
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
      <th>time_interval</th>
      <th>3377906280028510514</th>
      <th>3377906280395510514</th>
      <th>3377906281518510514</th>
      <th>3377906281774510514</th>
      <th>3377906282328510514</th>
      <th>3377906282418510514</th>
      <th>3377906283328510514</th>
      <th>3377906284028510514</th>
      <th>3377906284395510514</th>
      <th>...</th>
      <th>9377906282776510514</th>
      <th>9377906283125510514</th>
      <th>9377906283776510514</th>
      <th>9377906284555510514</th>
      <th>9377906285566510514</th>
      <th>9377906285615510514</th>
      <th>9377906286566510514</th>
      <th>9377906286615510514</th>
      <th>9377906288175510514</th>
      <th>9377906289175510514</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>183619</th>
      <td>2017-07-31 17:50:00</td>
      <td>NaN</td>
      <td>18.6</td>
      <td>4.5</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>3.9</td>
      <td>NaN</td>
      <td>21.4</td>
      <td>3.4</td>
      <td>...</td>
      <td>1.6</td>
      <td>18.7</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>NaN</td>
      <td>66.7</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>2.9</td>
    </tr>
    <tr>
      <th>183620</th>
      <td>2017-07-31 17:52:00</td>
      <td>NaN</td>
      <td>21.3</td>
      <td>6.1</td>
      <td>11.6</td>
      <td>NaN</td>
      <td>3.7</td>
      <td>NaN</td>
      <td>26.7</td>
      <td>3.6</td>
      <td>...</td>
      <td>2.0</td>
      <td>19.2</td>
      <td>12.6</td>
      <td>NaN</td>
      <td>16.9</td>
      <td>NaN</td>
      <td>66.9</td>
      <td>NaN</td>
      <td>12.0</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>183621</th>
      <td>2017-07-31 17:54:00</td>
      <td>NaN</td>
      <td>21.4</td>
      <td>8.5</td>
      <td>9.2</td>
      <td>NaN</td>
      <td>4.8</td>
      <td>NaN</td>
      <td>28.9</td>
      <td>3.9</td>
      <td>...</td>
      <td>2.9</td>
      <td>22.4</td>
      <td>15.0</td>
      <td>NaN</td>
      <td>15.7</td>
      <td>NaN</td>
      <td>55.0</td>
      <td>NaN</td>
      <td>10.3</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>183622</th>
      <td>2017-07-31 17:56:00</td>
      <td>NaN</td>
      <td>21.5</td>
      <td>7.7</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.1</td>
      <td>NaN</td>
      <td>24.6</td>
      <td>3.5</td>
      <td>...</td>
      <td>2.9</td>
      <td>19.5</td>
      <td>13.4</td>
      <td>NaN</td>
      <td>15.5</td>
      <td>NaN</td>
      <td>47.9</td>
      <td>NaN</td>
      <td>10.1</td>
      <td>3.7</td>
    </tr>
    <tr>
      <th>183623</th>
      <td>2017-07-31 17:58:00</td>
      <td>NaN</td>
      <td>24.0</td>
      <td>4.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.2</td>
      <td>NaN</td>
      <td>27.3</td>
      <td>3.4</td>
      <td>...</td>
      <td>2.5</td>
      <td>24.0</td>
      <td>15.3</td>
      <td>NaN</td>
      <td>16.1</td>
      <td>NaN</td>
      <td>42.1</td>
      <td>NaN</td>
      <td>11.1</td>
      <td>3.9</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 133 columns</p>
</div>




```python
travel_time['time_interval'].dropna(axis=0, how='any', inplace=True)
```

### 迁移车流量处理


```python
def process_car_migration(data_path, city_name):
    filename = os.path.join(data_path, "city_%s" % city_name, "migration.csv")
    migration = pd.read_csv(filename, 
                            sep=',', 
                            header=None,
                            names=['date', 's_city', 'e_city', city_name])

    # only use moving in "city" data, ignore moving out data
    df = migration[migration.e_city == city_name]
    df = df[["date", city_name]]

    # calculate total move in data of "city"
    df = df.groupby('date')[city_name].sum().reset_index()
    return df
```

# STGCN：时空图卷积网络
> 相关论文：[Spatio-Temporal Graph Convolutional Network \(STGCN\)](https://arxiv.org/pdf/1709.04875.pdf) 
在PGL中，提供了使用STGCN进行新冠疫情感染者趋势预测的示例。
## 数据集
需要将数据集按照下面的格式进行整理：
- input.csv: 历史迁移数据 with shape of [num\_time\_steps * num\_cities].

-  output.csv: 新增确诊数据 with shape of [num\_time\_steps * num\_cities].

- W.csv: 权重邻接矩阵 with shape of [num\_cities * num\_cities].

- city.csv: 城市列表.

### 开始训练

使用GPU训练示例
```
python main.py --use_cuda --input_file dataset/input_csv --label_file dataset/output.csv --adj_mat_file dataset/W.csv --city_file dataset/city.csv 
```

## 超参数

- n\_route: Number of city.
- n\_his: "n\_his" time steps of previous observations of historical immigration records.
- n\_pred: Next "n\_pred" time steps of New confirmed patients records.
- Ks: Number of GCN layers.
- Kt: Kernel size of temporal convolution.
- use\_cuda: Use gpu if assign use\_cuda. 

# 小结

## 总体感觉

应该说STGCN在真实场景的应用效果相当存疑，无论是传染病传播趋势预测还是交通流量预测，尽管论文一直在强调长期预测的准确性，但是估计还不能和机器学习算法相提并论，因为它本质上是RNN，RNN实际用的时候是很有问题的。

## 价值

个人感觉STGCN的几个关键价值在于大规模、时空域、交互关系，由于图计算和大数据高度相关，在大数据场景上，能够提供的解决方案还是很重要的。自回归再准，可解释性一直是短板，STGCN提高了可解释性，估计在学术上还是很有优势的。

另一方面，比如传染性传播预测，初赛要处理[118,30,135,75,34]个区域的预测，Prophet直接做到傻眼，只能搞一个不太准的批量预测（然后结果还是比官方baseline好emmmmm）。

## PGL上的STGCN

1. demo的数据样式没给，而且感觉demo迁移效果不好，只是对传染性传播、交通预测有用
2. 路网的模型跑不通，只好自己魔改了先
3. output的输出结果很奇怪，是归一化之后没还原回去还是咋地？

## 收获

目前跑通了两种场景

- 交通流量预测的自回归预测
  - 邻接矩阵W
  - 节点车速V（大宽表）
- 传染病传播预测
  - 邻接矩阵W（类似各地交流比值）
  - 人群迁移（大宽表）
  - label感染人数（大宽表）