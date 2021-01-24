---
author: 
date: 2021-01-24 16:50+08:00
layout: post
title: "目标检测-TTFNet"
description: ""
mathjax: true
categories:
- 目标检测
tags:
- TTFNet
typora-root-url: ..
---

* content
{:toc}
# TTFNet

论文:[Training-Time-Friendly Network for Real-Time Object Detection](https://arxiv.org/abs/1909.00700) 

代码链接: https://github.com/ZJULearning/ttfnet

**主要贡献**

本文就是针对**CenterNet**的小问题进行了改进，**主要贡献就是极大的减少了训练时间，大概减少了7倍时长**。

其核心思想是：**从BBOX框中编码更多的训练样本，主要是增加高质量正样本数，与增加批量大小具有相似的作用，这有助于扩大学习速度并加快训练过程。**主要是通过高斯核函数实现。其实很简单，我们知道centernet的回归分支仅仅在中心点位置计算loss，其余位置忽略，这种做法就是本文说的训练样本太少了，导致收敛很慢，需要特别长的训练时间，而本文采用高斯核编码形式，对中心点范围内满足高斯分布范围的点都当做正样本进行训练，可以极大的加速收敛。本文做法和FCOS非常类似，但是没有多尺度输出，也没有FPN等复杂结构。

作者首先做了一些证明：**提供更多高质量的训练样本与增加批次大小具有类似的作用，可以提供更多的监督信号，从而加速收敛。**

## 网络结构

和centernet差不多，使用了ResNet和DarkNet作为骨干，也是上采样到输入的1/4倍作为输出，区别是：为了减少小物体在下采样的时候丢失特征，引入了额外的挑层连接，类似Unet，假设高低层特征融合，可以有效缓解小物体精度较低的不足。

作者觉去掉了offset回归分支。目前只有两个分支输出，localization分支用于表示哪些区域有物体，和centernet里面含义相同，shape=(h/4,w/4,num_cls)。另一个是Regression分支，用于回归wh高，在centernet中，宽高回归分支的目标是基于中心点位置回归上距离，**由于这种简单的设置，导致其只能在中心点位置才能计算loss**，本文目的是增加正样本数，那么就不能采用这种目标编码方式了，本文采用的是FCOS的格式即回归的target是当前位置相对gt bbox的4条边的距离，故输出shape是(h/4,w/4,4)

![](/assets/objectdetection/img/7/ttf-1.png)

## 正负样本定义

![](/assets/objectdetection/img/7/ttf-2.png)

上图是一个直观的理解。假设图片中存在两个有交集的bbox，(a)是将bbox内部的所有区域都当做训练样本(一般就是值正样本)，这种策略属于早期论文做法，现在基本不会这么做了；(b)是带shrink操作的样本定义方式，考虑bbox标注的不准确性，只考虑内部一定范围内的区域是训练样本，大部分anchor-free论文都是这种做法；(c)是centernet特有的做法，仅仅在中心点处才是训练样本，其余位置忽略；(d)是本文基于高斯核来定义确定训练样本区域。一眼看起来比其他三种更加科学。

**(1) Object Localization** 
 这个其实是分类分支，shape=(h/4,w/4,num_cls)，label生成过程和centernet差不多，只不过centernet的高斯核是正方形核，而本文是椭圆形。

**(2) Size Regression** 
 其target不是和centernet一样，而是预测到4条边的距离，shape是(h/4,w/4,4)。训练样本区域和Object Localization一样，高斯核范围内的就是了。

![](/assets/objectdetection/img/7/ttf-3.png)

(ir,jr)是样本区域内的任意一点坐标，(w,h)是缩放了s倍，定义在特征图尺度的bbox的宽高值，对于任何一点其预测的bbox位置可以采用上述式子确定，也就是说，对于任意一点(ir,jr)，网络预测输出是(wl,ht,wr,hb)，注意这4个值是原图尺度，而不是特征图尺度哦！代表距离4条边的距离。这里的s不是stride=4,而是作者设置的16，目的是加快收敛。**centernet回归wh时候是没有归一化的，本文也没有进行归一化，但是为了方便优化，对宽高预测分支乘上了s=16，缩小预测范围，训练更加稳定一些。有些算法会将这个s设置为可学习参数，目的都是一样的。**

和FOCS一样，对于多个bbox重叠区域，其target是取最小bbox的标注。

## Loss计算

该分支采用的回归loss是最新的GIOU， 
 ![image.png-15.5kB](/assets/objectdetection/img/7/ttf-4.png)

B是原图尺度的bbox坐标(xmin,ymin,xmax,ymax)，是基于预测值进行解码还原后的bbox预测坐标。为了体现距离中心点远近，对loss的影响，引入了W权重。这个和FCOS的centerness分支左右差不多。

![image.png-25.9kB](/assets/objectdetection/img/7/ttf-5.png)

A是shrink后的矩形面积，G是前面得到的高斯核函数分布值，不考虑log(a)这一项，则就是简单的归一化加权而已，远距离的位置权重小，中心点附近权重大。而**乘上log(a)则可以反映出bbox的大小属性了**。

总的loss包括两个，一个是Object Localization，一个是Size Regression，权重设置比例为1:5。
