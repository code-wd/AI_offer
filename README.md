# AI_offer【算法面试经典】
> 这个项目主要用来总结**算法工程师面试**中常见的问题，期待能够帮助算法工程师们收获心仪的 offer

注意：GitHub 本身不支持 LaTeX 公式，如果要在线看的话可以安装插件 [TeX All the Things](https://chrome.google.com/webstore/detail/tex-all-the-things/cbimabofgmfdkicghcadidpemeenbffn?utm_source=chrome-ntp-icon)，刷新之后就可以正常显示公式；推荐本地使用 [Typora](https://typora.io/) 阅读，体验更好



# 目录

### 一、通用机器学习

- [x] 机器学习任务攻略
- [x] 正则化（L1，L2）
- [ ] Normalization
  - BN，Layer Normalization，Instance Normalization，Group Normalization 的作用、实现、相互间区别、使用场景【[参考1](https://www.nowcoder.com/discuss/714316?source_id=discuss_experience_nctrack&channel=-1)，[参考地址2](https://zhuanlan.zhihu.com/p/43200897)】
- [x] 优化算法
  - 不同优化算法：SGD、SGDM、NAG、AdaGrad、RMSprop、Adam、NAdam、AdaMax、AMSGrad
  - 优化算法效果可视化
- [x] 损失函数
  - **MAE, MSE, Smooth L1 Loss**, Huber Loss, Log-Cosh, Quantile, Hinge Loss, **交叉熵损失, Focal Loss**
  - 分类 Loss 为什么用交叉熵不用 MSE
- [x] 评估指标
  - 分类任务：准确率（Accuracy）、精确率（Precision）、召回率（Recall）、PR 曲线、F1 值、ROC 曲线、AUC、对数损失
  - 回归任务：平均绝对误差（MAE）、均方误差（MSE）、均方根误差（RMSE）、归一化均方根误差（NRMSE）、决定系数（R2）
  - 聚类任务：纯度、NMI、兰德系数、调整兰德系数
  - 目标检测任务：IOU
- [x] 激活函数
  - Sigmoid，Tanh，ReLU，Leaky ReLU，ELU，GELU
- [x] 梯度消失与梯度爆炸
  - 解释原因
  - 解决方案：合适的激活函数、Batch Norm、网络结构（ResNet，LSTM）、梯度裁剪
- [x] 过拟合与欠拟合
  - 什么是过拟合与欠拟合？解决方案是什么？
  - Dropout具体实现，训练和测试时有什么不一样？
- [ ] 其他小众问题
  - 偏差与方差
- [ ] 参数初始化方法
  - 权重初始化的方案（Xavier，kaiming_normal）
  - 神经网络权重初始化为零带来的影响
- [x] 模型压缩
- [ ] 调参技巧【[参考地址](https://www.zhihu.com/question/29641737/answer/243982984)】



### 二、机器学习

- [ ] 决策树【[参考地址](https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247486331&idx=1&sn=abc69ee44d932dd7d6bc4bbef82045c8&chksm=97ea211ea09da808a546c4a6f485f45289fea2e69cddc95aaf16f06ddfb9c428ecf9d259c980&scene=0&key=f7bb43d4492422e0472e06f4faf0076f9d9de975e8a73050e15cd63f1f549a060f9018009aa9f1f5f19aa37f1408ecb3ea2be5b8464b4eae89884e1d881c91ebef20c84ea9198fed470f36016f54c30a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=fLkIwHFJHv2%2FbP8RGzzYjXOZFDZBSwG0jwcmIEQhVi6CQoLDsa0PiCF8xoyKnPtI)】
  - ID3，C4.5，CART决策树的计算流程
  - 信息增益
  - 包括剪枝，预剪枝后剪枝好处坏处
- [ ] 集成学习
  - Adaboost，GBDT，XGboost，随机森林、lightgbm，bagging
  - GBDT防止过拟合的方法有哪些？怎么处理连续变量和离散变量？
  - 随机森林怎么处理样本不均衡
  - xgboost和gdbt怎么做回归和分类的？有什么区别？
- [ ] SVM 
  - 推导
  - SVM原理，为什么要用对偶问题解
  - SVM如何处理非线性情况（非核方法）软间隔和正则化
  - SVM的原理，核函数的有什么特点
  - LR和SVM的异同
- [ ] 判别式与生成式模型
- [ ] HMM是什么
- [ ] 牛顿法和拟牛顿法介绍
  - [ ] [二阶优化方法](https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247486331&idx=1&sn=abc69ee44d932dd7d6bc4bbef82045c8&chksm=97ea211ea09da808a546c4a6f485f45289fea2e69cddc95aaf16f06ddfb9c428ecf9d259c980&scene=0&key=f7bb43d4492422e0472e06f4faf0076f9d9de975e8a73050e15cd63f1f549a060f9018009aa9f1f5f19aa37f1408ecb3ea2be5b8464b4eae89884e1d881c91ebef20c84ea9198fed470f36016f54c30a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=fLkIwHFJHv2%2FbP8RGzzYjXOZFDZBSwG0jwcmIEQhVi6CQoLDsa0PiCF8xoyKnPtI)，相比一阶的区别
- [ ] 无监督算法
  - 基于密度的算法
  - k-means 
- [ ] 逻辑回归
  - 基本概念
  - LR的原理，手写损失，手写反向传播
  - LR特征重复会怎么样
- [ ] 朴素贝叶斯的推导以及假设条件
- [ ] KNN
  - KNN的原理，如果所有K选的很大，会导致过拟合还是欠拟合
- [ ] DETR算法
- [ ] 问对NAS算法
- [ ] LGBM和XGB区别
- [ ] Inception系列的演化
- [ ] 谈谈传统不平衡分类算法（上采样、下采样、混合采样、代价敏感矩阵、调整分类器阈值）以及最新深度学习不平衡分类算法



### 三、数据相关

- [ ] 样本不平衡处理方法
  - 样本不平衡怎样选择特征
- [ ] 数据增强的常用方案
- [ ] 特征工程【[参考地址](https://github.com/wangyuGithub01/Machine_Learning_Resources)】
  - 怎样选择特征
  - 特征工程做的有哪些？非线性可分的情况怎么处理的？
  - 标准化、归一化、异常特征清洗、不平衡数据
  - 不平衡数据的处理方法
- [ ] 数据预处理
- [ ] 稀疏特征的处理
- [ ] 如何识别数据集中的噪声样本



### 四、计算机视觉

- [ ] RCNN 系列算法的演进【[参考地址](https://www.nowcoder.com/discuss/762721?source_id=discuss_experience_nctrack&channel=-1)】【[地址2](https://zhuanlan.zhihu.com/p/58776542)】【[地址3](https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247486331&idx=1&sn=abc69ee44d932dd7d6bc4bbef82045c8&chksm=97ea211ea09da808a546c4a6f485f45289fea2e69cddc95aaf16f06ddfb9c428ecf9d259c980&scene=0&key=f7bb43d4492422e0472e06f4faf0076f9d9de975e8a73050e15cd63f1f549a060f9018009aa9f1f5f19aa37f1408ecb3ea2be5b8464b4eae89884e1d881c91ebef20c84ea9198fed470f36016f54c30a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=fLkIwHFJHv2%2FbP8RGzzYjXOZFDZBSwG0jwcmIEQhVi6CQoLDsa0PiCF8xoyKnPtI)】
  - faster rcnn的最后输出大小是多少、它的正负样本是怎么选择
  - 目标检测中如何解决目标尺度大小不一的情况）+基础知识（pooling反向传播，python拷贝文件）
- [ ] YOLO 系列算法的演进
- [ ] anchor-free 的目标检测算法
- [ ] 卷积
  - 普通卷积，Group卷积，深度可分离卷积实现与参数量计算
  - 转置卷积、空洞卷积、1*1 卷积（变换channel，通道融合，降维）、deformable conv
  - 感受野的概念
- [ ] ROI pooling 与 ROI align【[地址](https://zhuanlan.zhihu.com/p/85035860)】
- [ ] IOU
- [ ] 非极大值抑制（NMS）
- [ ] 目标检测，任意形状的检测框应该怎样实现
- [ ] Transformer相比较CNN的优缺点
- [ ] FPN为何能够提升小目标的精度
- [ ] resnet和densenet及其不同
- [ ] [计算机视觉四大基本任务](https://zhuanlan.zhihu.com/p/58776542)
- [ ] [pooling层的作用](https://blog.csdn.net/sunflower_sara/article/details/81322048)
- [ ] 



### 五、自然语言处理

- [ ] word2vec和onehot的区别
  - 讲讲word2vec原理，介绍一下tf-idf
  - WORD2VEC的细节（CBOW+skop-gram huffman树+负采样算
  - 介绍word2vec，word2vec假设，负采样，参数数目

- [ ] RNN 与 LSTM，Bi-LSTM，GRU 原理
  - lstm原理三个门作用
  - LSTM结构画图，为什么能解决梯度消失和梯度爆炸

- [ ] transformer 与 BERT
  - self-attention 原理
  - transformer 原理与优缺点
  - BERT 原理【[参考地址](https://www.nowcoder.com/discuss/714316?source_id=discuss_experience_nctrack&channel=-1)】
  - 位置信息，bert 与 transformer 的位置信息来源不同之处【为什么 transformer 用 positional encodding 而 bert 自己学习 positional embedding】
  - bert 的变种模型
  - bert的embedding是相加还是concat？
  - bert的根号dk作用【dot production 为什么要除以维度的开根号】
  - 多头注意力原理与作用
  - 描述下多头自注意力机制，自注意力公式，说说为什么用LN，为什么用残差结构
- [ ] LSTM+attention这个模型为什么要在加attention



### 六、手撕代码

> 这一部分不会涉及一般的算法，只会列出设计机器学习的算法

- [ ] 手写img2col
- [ ] nms
- [ ] IOU
- [ ] pytorch 实现 逻辑回归
- [ ] k-means
- [ ] pytorch 实现多头注意力机制
- [ ] logistic回归
- [ ] 快速排序
- [ ] 梯度下降算法推导
  - 反向传播【[参考地址](https://mp.weixin.qq.com/s?__biz=MzIxODM4MjA5MA==&mid=2247486331&idx=1&sn=abc69ee44d932dd7d6bc4bbef82045c8&chksm=97ea211ea09da808a546c4a6f485f45289fea2e69cddc95aaf16f06ddfb9c428ecf9d259c980&scene=0&key=f7bb43d4492422e0472e06f4faf0076f9d9de975e8a73050e15cd63f1f549a060f9018009aa9f1f5f19aa37f1408ecb3ea2be5b8464b4eae89884e1d881c91ebef20c84ea9198fed470f36016f54c30a&ascene=14&uin=MTM2NDUyMTkxOQ%3D%3D&devicetype=Windows+10&version=62060739&lang=zh_CN&pass_ticket=fLkIwHFJHv2%2FbP8RGzzYjXOZFDZBSwG0jwcmIEQhVi6CQoLDsa0PiCF8xoyKnPtI)】



### 七、Python 语言

- [ ] python多线程，多进程
- [ ] python的GIL
- [ ] 生成器与迭代器
- [ ] 装饰器
- [ ] 深浅拷贝
- [ ] python 的垃圾回收机制
- [ ] pyhton的 is 和 == 有什么区别
- [ ] python的class在继承父类时，super的作用是什么
- [ ] list  参数传递

https://github.com/code-wd/
