---
author: 
date: 2021-01-24 15:20+08:00
layout: post
title: "目标检测-CenterNet"
description: ""
mathjax: true
categories:
- 目标检测
tags:
- CenterNet
typora-root-url: ..
---

* content
{:toc}
# CenterNet

论文名称：[Objects as Points](https://arxiv.org/pdf/1904.07850.pdf)

代码链接：https://github.com/xingyizhou/CenterNet

**主要贡献**

构建模型时将目标作为一个点——即目标BBox的中心点。我们的检测器采用关键点估计来找到中心点，并回归到其他目标属性，例如尺寸，3D位置，方向，甚至姿态。

第一，我们分配的锚点仅仅是放在位置上，没有尺寸框。没有手动设置的阈值做前后景分类。（像Faster RCNN会将与GT IOU >0.7的作为前景，<0.3的作为背景，其他不管）；

第二，每个目标仅仅有一个正的锚点，因此不会用到NMS，我们提取关键点特征图上局部峰值点（local peaks）；

第三，CenterNet 相比较传统目标检测而言（缩放16倍尺度），使用更大分辨率的输出特征图（缩放了4倍），因此无需用到多重特征图锚点；

## 网络结构

**backbone**

骨架网络采用了4种，分别是ResNet-18, ResNet-101, DLA-34和Hourglass-104。

**head**

对于目标检测而言，输出三个特征图，分别是高斯热图(h/4,w/4,cls_nums)，每个通道代表一个类别；宽高输出图(h/4,w/4,2)，代表对应中心点处预测的宽高；中心点量化偏移图(h/4,w/4,2)，这个分支对于目标检测而言可能作用不会很大，因为我们最终要的是bbox,而不是中心点，但是对于姿态估计而言就非常重要了。

## Loss计算

$$
L=L_{k}+\lambda_{size} L_{size}+\lambda_{off} L_{off}
$$



**heatmap部分损失函数：**和cornernet一样GaussianFocalLoss，只是生成heatmap的高斯中心不再是角点，而是gt中心，高斯半径r计算和cornernet一样，按道理是通过iou约束计算到的r，角点和中心计算方式不同，但作者认为影响不大。后续TTFNet算法有改进。


$$
L_{k}=\frac{-1}{N}\sum_{c=1}^{C}\sum_{i=1}^{H}\sum_{j=1}^{W}\left \{ \begin{array}{c} (1-p_{cij})^{\alpha}\log(p_{cij}) & \text{if }  y_{cij}=1 \\ (1-y_{cij})^{\beta}(p_{cij})^{\alpha}\log(1-p_{cij}) & \text{otherwise} \end{array} \right.
$$


**offset部分损失函数：**和cornernet一样，是表示在取整计算时丢失的精度信息，但这里使用的L1损失。$ \hat{O_\tilde{p}}$是预测的offset，$(\frac{p}{R}-\tilde{p})$是实际offset。


$$
L_{off}=\frac{1}{N}\sum_{p}\left | \hat{O_\tilde{p}}-(\frac{p}{R}-\tilde{p})\right|
$$



**size部分损失函数：**$s_k$是gt下采样的长宽，也是使用的L1损失


$$
L_{size}=\frac{1}{N}\sum_{k=1}^{N}\left | \hat{S_{p_{k}}}-s_k\right|
$$



论文中$\lambda_{size}=0.1$，$\lambda_{off}=1$

head输出[80x128x128], [2x128x128], [2x128x128]

## 推理流程

<img src="/assets/objectdetection/img/6/centernet-1.png" style="zoom:50%;" />

最终网络在每个位置预测c+4个输出，c是类别数，4则包含横向和纵向的offset以及宽高。前向阶段，对于每个类别的热度图，提取峰值点，若一个点的值大于等于其周围八个点的值（通过3x3maxpool），则认为是峰值点，选取top100个峰值点，代码中采用0.3作为阈值对这些峰值点进行筛选。bbox坐标转换：


$$
(\hat{x_i}+\sigma\hat{x_i}-\hat{w_i}/2,\hat{y_i}+\sigma\hat{y_i}-\hat{h_i}/2,\\\hat{x_i}+\sigma\hat{x_i}+\hat{w_i}/2,\hat{y_i}+\sigma\hat{y_i}+\hat{h_i}/2)\\
(\sigma\hat{x_i},\sigma\hat{y_i})=\hat{O}_{\hat{x_i},\hat{y_i}}\\
(\hat{w_i},\hat{h_i})=\hat{S}_{\hat{x_i},\hat{y_i}}
$$

