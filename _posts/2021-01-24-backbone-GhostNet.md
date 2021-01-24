---
author: 
date: 2021-01-24 12:52+08:00
layout: post
title: "Backbone-GhostNet"
description: ""
mathjax: true
categories:
- Backbone
tags:
- 轻量级
- GhostNet
typora-root-url: ..
---

* content
{:toc}
# GhostNet

论文名称：[GhostNet: More Features from Cheap Operations](https://arxiv.org/abs/1911.11907)

GitHub：https://github.com/huawei-noah/ghostnet

**主要贡献**

从特征图冗余问题出发，提出一个仅通过少量计算就能生成大量特征图的结构Ghost Module。

## 算法设计

在优秀CNN模型中，特征图存在冗余是非常重要的，但是很少有人在模型结构设计上考虑特征图冗余问题（The redundancy in feature maps）。

而本文就从特征图冗余问题出发，提出Ghost Module仅通过少量计算（*cheap operations*）就能生成大量特征图。经过线性操作生成的特征图称为ghost feature maps，而被操作的特征图称为intrinsic feature maps。

**问题1: 何为特征图冗余？**

<img src="/assets/classification/img/8/ghostnet-1.jpeg" style="zoom:50%;" />

看着相似的那些就是冗余,作者用红绿蓝重点给我们标记的那些就是冗余特征图的代表

**问题2: Ghost feature maps 和 Intrinsic feature maps 是什么？**

<img src="/assets/classification/img/8/ghostnet-2.jpeg" style="zoom: 25%;" />

一组特征图中，一部分是Intrinic，而另外一部分是可以由 intrinsic 通过cheap operations来生成的。

**问题3: Linear transformations 和 Cheap operations 是什么？**

 linear operations 等价于 cheap operations即是 诸如3 * 3的卷积，或者5 * 5的卷积。

**问题4: Ghost Module长什么样？Ghost Bottlenecks长什么样？Ghost Net长什么样？**

<img src="/assets/classification/img/8/ghostnet-3.jpeg" style="zoom:50%;" />

通常的卷积如图2（a）所示，而Ghost Module则分为两步操作来获得与普通卷积一样数量的特征图（这里需要强调，是数量一样）。

第一步：少量卷积（比如正常用32个卷积核，这里就用16个，从而减少一半的计算量）

第二步：cheap operations，如图中的Φ表示，从问题3中可知，Φ是诸如3*3的卷积，并且是逐个特征图的进行卷积（Depth-wise convolutional）。

这里应该是本文最大的创新点和贡献了。

了解了Ghost Module，下面看Ghost Bottlenecks。

<img src="/assets/classification/img/8/ghostnet-4.png" style="zoom: 50%;" />

结构与ResNet的是类似的，并且与mobilenet-v2一样在第二个module之后不采用ReLU激活函数。

左边是stride=1的Ghost Bottlenecks，右边是stride=2的Ghost Bottlenecks，目的是为了缩减特征图大小。

接着来看Ghost Net，Ghost Net结构与MobileNet-V3类似，并且用了SE结构，如下表，其中#exp表示G-bneck的第一个G-Module输出特征图数量

<img src="/assets/classification/img/8/ghostnet-5.jpeg" style="zoom: 25%;" />

## 总结

Ghost  Module的想法很巧妙，可即插即用的实现轻量级卷积模型，但若能实现不训练的轻量级卷积模型，那就更好了。这也是本笔记中遗憾的部分，未能实现不训练的即插即用，在此希望集思广益，改进上述提出的不成熟方案，说不定GhostNet-V2就诞生了，当然更期待原作者提出GhostNet-V2。