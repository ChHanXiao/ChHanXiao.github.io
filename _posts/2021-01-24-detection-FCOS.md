---
author: 
date: 2021-01-24 14:26+08:00
layout: post
title: "FCOS"
description: ""
mathjax: true
categories:
- 目标检测
tags:
- FCOS
typora-root-url: ..
---

* content
{:toc}
# FCOS

论文名称： [FCOS: A simple and strong anchor-free object detector](https://arxiv.org/pdf/2006.09214v3.pdf )

论文名称： [FCOS: Fully Convolutional One-Stage Object Detection](https://arxiv.org/pdf/1904.01355v1.pdf) 

**主要贡献**

FCOS是目前最经典优雅的一阶段anchor-free目标检测算法，其模型结构主流、设计思路清晰、超参极少和不错的性能使其成为后续各个改进算法的baseline，和retinanet一样影响深远。

anchor-base的缺点是：**超参太多，特别是anchor的设置对结果影响很大,不同项目这些超参都需要根据经验来确定，难度较大。** 而anchor-free做法虽然还是有超参，但是至少去掉了anchor设置这个最大难题。fcos算法可以认为是point-base类算法也就是特征图上面每一个点都进行分类和回归预测，简单来说就是anchor个数为1的且为正方形anchor-base类算法。

## 网络结构

![](/assets/objectdetection/img/4/fcos-1.png)

fcos和retinanet网络相似

**backbone**

retinanet在得到p6,p7的时候是采用c5层特征进行maxpool得到的，而fcos是从p5层抽取得到的，而且其p6和p7进行卷积前，还会经过relu操作，retinanet的FPN没有这个算子(**C5不需要是因为resnet输出最后就是relu**)。

**head**

和retinanet相比，fcos的head结构多了一个centerness分支，其余也是比较重量级的两条不共享参数的4层卷积操作，然后得到分类和回归两条分支。fcos是point-base类算法，对于特征图上面任何一点都回归其距离bbox的4条边距离，由于大小bbox且数值问题，一般都会对值进行变换，也就是除以s，主要目的是**压缩预测范围，容易平衡分类和回归Loss权重**。如果某个特征图上面点处于多个bbox重叠位置，则该point负责小bbox的预测。

![](/assets/objectdetection/img/4/fcos-2.png)

## 正负样本定义

(1) 对于任何一个gt bbox，首先映射到每一个输出层，利用center_sampling_ratio值计算出该gt bbox在每一层的正样本区域以及对应的left/top/right/bottom的target 
(2) 对于每个输出层的正样本区域，遍历每个point位置，计算其max(left/top/right/bottom的target)值是否在指定范围内regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),(512, INF)，不再范围内的认为是背景

**CNN对尺度是非常敏感的，一个层负责处理各种尺度，难度比较大，采用FPN来限制回归范围可以减少训练难度** 

## loss计算

对于分类分支其采用的是FocalLoss，参数和retinanet一样。对于回归分支其采用的是GIou loss。在这种设置情况下，作者发现训推理时候会出现一些奇怪的bbox，原因是对于回归分支，正样本区域的权重是一样的，同等对待，导致那些虽然是正样本但是离gt bbox中心比较远的点对最终loss产生了比较大的影响，其实这个现象很容易想到，但是解决办法有多种。作者解决办法是引入额外的centerness分类分支，该分支和bbox回归分支共享权重，仅仅在bbox head最后并行一个centerness分支，其target的设置是离gt bbox中心点越近，该值越大，范围是0-1。虽然这是一个回归问题，但是作者采用的依然是ce loss。

越靠近中心,min(l,r)和max(l,r)越接近1，也就是越大。

![](/assets/objectdetection/img/4/fcos-3.png)

## 附加内容

fcos代码在第一版和最终版上面修改了很多训练技巧，对最终mAP有比较大的影响，主要是：

**pr(1) centerness 分支的位置** 
 早先是和分类分支放一起，后来和回归分支放一起 
 **(2) 中心采样策略** 
 早先版本是没有中心采样策略的 
 **(3) bbox预测范围** 
 早先是采用exp进行映射，后来改成了对预测值进行relu，然后利用scale因子缩放 
 **(4) bbox loss权重** 
 早先是所有正样本都是同样权重，后来将样本点对应的centerness target作为权重，离GT中心越近，权重越大 
 **(5) bbox Loss选择** 
 早先采用的是iou，后面有更优秀的giou，或许还可以尝试ciou 
 **(6) nms阈值选取** 
 早先版本采用的是0.5，后面改为0.6