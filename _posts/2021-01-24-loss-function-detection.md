---
author: 
date: 2021-01-24 09:52+08:00
layout: post
title: "损失函数-目标检测"
description: ""
mathjax: true
categories:
- Loss
tags:
- Smooth L1
- IoU
- GIoU
- DIoU
- CIoU
typora-root-url: ..
---

* content
{:toc}

# 目标检测

目标检测任务的损失函数一般由Classificition Loss（分类损失函数）和Bounding Box Regeression Loss（回归损失函数）两部分构成。Bounding Box Regeression的Loss近些年的发展过程是：Smooth L1 Loss-> IoU Loss（2016）-> GIoU Loss（2019）-> DIoU Loss（2020）-> CIoU Loss（2020）

好的目标框回归函数应该考虑三个重要几何因素：**重叠面积**、**中心点距离**，**长宽比**。

IOU_Loss：主要考虑检测框和目标框重叠面积。

GIOU_Loss：在IOU的基础上，解决边界框不重合时的问题。

DIOU_Loss：在IOU和GIOU的基础上，考虑边界框中心点距离的信息。

CIOU_Loss：在DIOU的基础上，考虑边界框宽高比的尺度信息

## Smooth L1 Loss

Smooth L1 Loss能从两个方面限制梯度：当预测框与 ground truth 差别过大时，梯度值不至于过大；当预测框与 ground truth 差别很小时，梯度值足够小。


$$
\begin{align}
L_2(x) &= x^2  \\
L_1(x) &= x \\
smooth_{L_1}(x) &=\left \{ \begin{array}{c} 0.5x^2 & if \mid x \mid <1 \\ \mid x \mid - 0.5 & otherwise  \end{array} \right.
\end{align}
$$


损失对于$x$的导数为：


$$
\begin{align}
\frac{\partial L_2(x)}{\partial x} &= 2x \\
\frac{\partial L_1(x)}{\partial x} &= \left \{ \begin{array}{c} 1 & \text{if }  x \geq 0 \\ -1 & \text{otherwise} \end{array} \right. \\
\frac{\partial smooth_{L_1}(x)}{\partial x} &=\left \{ \begin{array}{c} x & if \mid x \mid <1 \\ \pm1 & otherwise  \end{array} \right.
\end{align}
$$


<img src="/assets/lossfunction/img/1/smoothl1-l1-l2.png" style="zoom:67%;" />

从损失函数对x的导数可知：L1损失函数对x的导数为常数，在训练后期，x很小时，如果learning rate 不变，损失函数会在稳定值附近波动，很难收敛到更高的精度。L2失函数对x的导数在x值很大时，其导数也非常大，在训练初期不稳定。 Smooth L1完美的避开了L1、L2​损失的缺点。

##  IoU Loss

Smooth L1 loss不能很好的衡量预测框与ground true 之间的关系，相对独立的处理坐标之间的关系可能出现Smooth L1 loss相同，但实际IoU不同的情况。因此，提出IoU loss，将四个点构成的box看成一个整体进行损失的衡量$L_{IoU}=-\ln IoU(A,B)$也可定义为$1-IoU(A,B)$。IoU loss具有尺度不变性，大边界框的IoU loss 基本上与小边界框的IoU loss相等，本质上是对IoU的交叉熵损失，即将IoU视为伯努利分布的随机采样。

<img src="/assets/lossfunction/img/1/IoU-1-1.png"  />

<img src="/assets/lossfunction/img/1/IoU-1-2.png"  />

可以看到IoU的loss其实很简单，主要是**交集/并集**，但其实也存在两个问题

**问题1**：即状态1的情况，当预测框和目标框不相交时，**IOU=0**，无法反应两个框距离的远近，此时损失函数不可导，IOU Loss无法优化两个框不相交的情况。

**问题2**：即状态2和状态3的情况，当两个预测框大小相同、IOU也相同，IOU Loss无法区分两者相交情况的不同。

## GIoU Loss

GIoU Loss中在原来的IOU损失的基础上加上一个惩罚项，就可以衡量预测框与真实框不相交的情况

![](/assets/lossfunction/img/1/GIoU-1-1.png)

![](/assets/lossfunction/img/1/GIoU-1-2.png)

GIoU是IoU的下界，当且仅当两个框完全重合时相等，IoU取值范围为[0,1]，GIoU取值范围为[-1,1]，IoU只关注重叠区域不同，GIoU不仅关注重叠区域，还关注其他的非重合区域，能更好的反映两者的重合度。

**问题**：状态1、2、3都是预测框在目标框内部且预测框大小一致的情况，这时预测框和目标框的差集都是相同的，因此这三种状态的**GIOU值**也都是相同的，这时GIOU退化成了IOU，无法区分相对位置关系。

## DIoU Loss

DIOU_Loss考虑了**重叠面积**和**中心点距离**。
$$
L_{DIoU}=1-(IoU-\frac{\rho^2(b,b^{gt})}{c^2})
$$
其中$b$,$b^{gt}$分别表示$B$,$B^{gt}$的中心点$\rho(·)$为欧氏距离，$c$为$B$,$B^{gt}$的最小外接矩形的对角线距离。

![](/assets/lossfunction/img/1/DIoU-1-1.png)

![](/assets/lossfunction/img/1/DIoU-1-2.png)

**问题**：比如上面三种状态，目标框包裹预测框，本来DIOU_Loss可以起作用。但预测框的中心点的位置都是一样的，因此按照DIOU_Loss的计算公式，三者的值都是相同的。

## CIoU Loss

CIOU_Loss和DIOU_Loss前面的公式都是一样的，不过在此基础上还增加了一个影响因子$\alpha\nu$，将预测框和目标框的长宽比都考虑了进去。
$$
L_{CIoU}=1-(IoU-\frac{\rho^2(b,b^{gt})}{c^2}-\alpha\nu)\\
\alpha=\frac{\nu}{(1-IoU)+\nu}\\
\nu=\frac{4}{\pi^2}(\arctan{\frac{w^{gt}}{h^{gt}}}-\arctan{\frac{w}{h}})^2
$$
$\nu$是用来衡量长宽比一致性的参数。

