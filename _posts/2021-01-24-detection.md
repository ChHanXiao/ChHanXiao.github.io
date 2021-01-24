---
author: 
date: 2021-01-24 12:52+08:00
layout: post
title: "目标检测问题"
description: ""
mathjax: true
categories:
- 目标检测
tags:
- 目标检测
---

* content
{:toc}

# 目标检测问题

## 建模方式

**分治**（multi-region classification & regression）

一张图上预定义若干regions，然后对每个region逐一进行**分类与回归**

这个region附近是否存在物体，如果存在，那么这一物体是什么类别，而它的位置相对于这个region又在哪里

## 需要解决的问题

这种建模方式需要解决哪些问题？

### 问题1：怎样设计“region”

这个“region”的设计更多的反映的是**对于所需要解决的问题及场景的先验**。

在RetinaNet等anchor-based的算法中，它的名字叫做**anchor**；在FCOS等anchor-free的算法中，它又代表着**anchor point**；

### 问题2：每个“region”负责多大的范围

概括为**anchor design** & **label assignment**问题

**RetinaNet**在图片上均匀平铺了一系列不同尺度、不同长宽比的region(anchor)，并基于gt bbox和anchor的IoU，定义了“管辖范围”；而**FCOS**等anchor-free算法中则平铺了一系列anchor point，并采用中心点距离+尺度约束的方式定义了“管辖范围”；**ATSS**中提出设置自适应的阈值，来定义“管辖范围”；当然上述方案都是静态的设定了“管辖范围”，近年出现了另一类改进方案，诸如**FreeAnchor，AutoAssign，PAA**等算法中则提出了动态的 “管辖范围”设置方案（即“管辖范围”与网络的输出相关）；

对于anchor design & label assignment的问题，如果想要得到更理想的检测算法性能，需要**研究dense和multi-stage refine之间的折中**，有些致力于解决极端正负样本不平衡问题以使得**dense anchor**发挥出更大的作用，例如RetinaNet (Focal loss)；有些则设计各种**多级refine**的方案，例如Faster RCNN，Cascade RCNN， RefineDet，RPDet，AlignDet，Guided anchoring，Cascade RPN等；有些则尝试提出更有效的**region & label assignment**方案，例如FCOS，CenterNet，ATSS，FreeAnchor等。

### 问题3：怎样有效提取每个“region”的特征做分类和回归

首先，我们可以考虑多个区域**共享由全图提取的特征**，而无需逐区域计算特征（Fast R-CNN中提出）；其次，CNN可以提取**“规范”区域**的特征，即落在**grid**上、**长宽比一定**的区域的特征；而对于**“非规范”区域**，可以通过池化/插值的形式得到这些区域的特征（ROI pooling/ROI Align/Deformable Conv w. specified offsets等）。从这个角度说，**如果把anchor理解为original proposal，那么faster r-cnn等一系列multi-stage detector中rpn和r-cnn所做的事情并没有本质差别。**

## 目标检测网络

可以从以下6部分分析一个目标检测网络

- 网络backbone设计
- 网络neck设计
- 网络head设计
- 正负样本定义
- bbox编解码设计
- loss设计

