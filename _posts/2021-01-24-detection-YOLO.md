---
author: 
date: 2021-01-24 13:52+08:00
layout: post
title: "YOLO系列"
description: ""
mathjax: true
categories:
- 目标检测
tags:
- YOLOv1
- YOLOv2
- YOLOv3
- YOLOv4
- YOLOv5
typora-root-url: ..
---

* content
{:toc}
# YOLO系列

[darknet](https://pjreddie.com/darknet/yolo/) 

YOLOv1论文：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)

YOLOv2论文：[YOLO9000:Better, Faster, Stronger](http://web.eng.tau.ac.il/deep_learn/wp-content/uploads/2018/01/YOLO9000.pdf)

YOLOv3论文：[Yolov3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)

YOLOv4论文：[Yolov4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/pdf/2004.10934.pdf)

来源参考

[mmdetection-mini](https://github.com/hhaAndroid/mmdetection-mini)

## YOLOv1

**主要贡献**：提出了一个极度简单、通用、思路清晰、速度和精度平衡的目标检测算法。不需要复杂的两阶段refine思想，一次性即可学习出图片中所有目标类别和坐标信息，算是one-stage目标检测的先行者。

### 网络结构

![](/assets/objectdetection/img/1/YOLOv1-1.png)

设计的骨架比较简单，就是标准卷积+池化的直筒结构。卷积层后面的激活函数是 Leaky ReLU 。

最后一层是采用fc形式输出，没有激活函数，shape为(batch,S×S×(B\*5+C)，S是网格参数，B是每个网格输出的bbox个数，C是类别数，5是表示bbox的xywh预测值加上一个置信度预测值。对于VOC数据且输入是448x448，S=7，B=2，则输出是(batch,7x7x(2*(4+1)+20))=(batch,7x7x30)。

前20个数是不包括背景的类别信息，值最大的索引就是对应类别；由于B=2，所以前2个数是置信度分支；后面2x4个数按照xywhxywh的格式存储。

置信度分值反映了模型对于这个网格的预测，包括两个信息：该网格是否含有物体，以及这个bbox的坐标预测有多准。值越大表示越可能有前景物体，且该网格的预测bbox越准确**。是否可以舍弃置信度预测值，只预测类别和位置？**答案是：如果仅仅考虑该网格是否含有物体的作用，那么完全可以舍弃，把这个功能并入分类分支即可(cls_num+1，1是背景类别)，但是如果还要同时考虑预测bbox准确度，那么就不能舍弃，否则无法实现这个功能。

在强行施加了网格限制以后，每个网格最多只能输出一个预测结果(不管B设置为多少都一样,需要知道正样本定义规则才能理解)，所以该算法最大不足是**理论召回率比较低**，特别是在物体和物体靠的比较近且物体比较小的时候，很大几率多个物体中心坐标会落在同一个网格中，导致漏检。

<img src="/assets/objectdetection/img/1/YOLOv1-2.png" style="zoom:50%;" />

### 正负样本定义

对于任何一个gt bbox，如果其中心坐标落在某个网格内部，那么该网格就负责预测该物体。

有一个细节需要注意：B=2也就是每个网格要预测两个bbox，将B个xywh预测值都进行解码还原为预测bbox格式，然后计算B个预测bbox和对应gt bbox的iou，哪个iou大就哪个xywh预测框负责预测该gt bbox，另一个当做负样本。那么如果某个网格有m(m>=2)个gt bbox中心落在里面，匹配规则是将B个预测bbox和所有gt  bbox计算iou，然后取最大iou就可以找到哪个预测框和哪个gt bbox最匹配，此时该预测框就是正样本，其余全部算负样本，其余gt  bbox被忽略当做背景处理。

### bbox编解码

理论上每个网格的xywh预测值可以是gt bbox的真实值，但是这样做不太好，**因为xy和wh的预测范围不一样，并且分类和bbox预测分支取值范围也不一样，如果不做任何设计会出现某个分支训练的好而其余分支训练的不好现象，效果可能不太好，**所以对bbox进行编解码是非常关键的。

对于xy预测值的target，其表示gt bbox中心相当于所负责网格左上角的偏移，范围是0～1。假设该gt bbox的中心坐标为(200,300)，图片大小是448x448，S=7，那么其xy label计算过程是：

1. 将gt bbox中心坐标值映射到特征图上，其实就是除以stride=448/7=64,得到(200/64=3.125,300/64=4.69)
2. 然后向下取整得到所负责该gt的网格坐标(3,4)
3. 计算xy值label=(0.125,0.69)

对于wh预测就直接将gt bbox的宽高除以图片宽高归一化即可，其范围也是0～1。这个操作本身没有问题，但是其忽略了大小bbox属性，会出现小物体梯度比较小的问题出现漏检，作者做法是采用平方根变换。其本质是压缩大小物体回归差距，使得大小不同尺度物体在loss层面相同时候更加重视小物体。

### loss设计

<img src="/assets/objectdetection/img/1/YOLOv1-3.png"  />

**所有loss都是L2 loss即都是回归问题**，并且因为是回归问题，所以即使xywh预测范围是0～1也没有采用sigmoid进行强制压缩范围(原因是sigmoid有饱和区)。

1. 对于xywh向量，其仅仅计算正样本loss，其余位置是忽略的
2. 对于类别向量，其也是仅仅计算正样本loss，其余位置是忽略的，label向量是One-hot编码
3. 对于置信度值，其需要考虑正负样本，并且正负样本权重不一样。特别的为了实现置信度值的功能：该网格是否含有物体，以及这个box的坐标预测的有多准，**对于正样本，其置信度的label不是1，而是预测bbox和gt bbox的iou值，对于负样本，其置信度label=0。**当然如果你希望置信度值仅仅具有该网格是否含有物体功能，那么只需要正样本label设置为1即可，代码实现上通过变量rescore控制。

因为正样本非常少，故其设置了$\lambda_{noobj}=0.5,\lambda_{coord}=5$，参数来平衡正负样本梯度。

### 推理流程

(1) 遍历每个网格，将B个置信度预测值和预测类别向量分值相乘，得到每个bbox的类相关置信度值，可以发现这个乘积即表示了预测的bbox属于某一类的概率和该bbox准确度的信息，此时一共有SXSXB个预测框 
 (2) 设置阈值，滤掉类相关置信度值低的boxes，同时将剩下的bbox还原到原图尺度**，xy中心坐标还原是首先加上当前网格左上角坐标，然后乘上stride即可，而wh值就直接乘上图片wh即可** 
 (3) 对剩下的boxes进行NMS处理，就得到最终检测结果

### 总结

YOLO可以说是第一个简洁优美的one-stage目标检测算法，设计思路非常简洁清晰，最大优点是速度极快、背景误检率低，缺点是由于网格特点召回率会低一些，回归精度也不是很高，特别是小物体，这些缺点在后续的YOLOv2-v3中都有极大改善。

## YOLOv2

**主要贡献**：对YOLOv1进行改进，大幅提升了定位准确度和召回率，同时速度也是极快。

**trick：**

- batch norm：BN的作用主要是加快收敛，避免过拟合


- new network：提出darknet19


- 全卷积形式：YOLOv1采用全连接层输出学习适应不同物体的形状比较困难(本身cnn就不具有尺度不变性)导致YOLOv1在精确定位方面表现较差。


- anchor bbox：n个anchor bbox，提高召回率，可以真正实现每个网格预测多个物体。 **但是训练策略依然采用的是YOLOv1中的，正负样本不平衡问题应该是加剧了导致mAP稍有下降。**


- dimension priors：anchor kmean自动聚类算法


- passthrough：在head部分所提的从高分辨率特征图变成低分辨率特征图的特征融合操作


ssd是完全基于iou阈值进行匹配，只要anchor设置和iou阈值设置合适就可以保证正样本比较多，会更稳定。而yolo匹配策略仅仅是每个gt bbox一定匹配唯一的anchor，不管你anchor设置的如何，其正样本非常少，前期训练肯定更加不稳定。

### 网络结构

**backbone**

引入BN层、新的骨架网络darknet19，每个convolutional都包括conv+bn+leakyrelu模块，和vgg相同的直筒结构。

**head**

假设网络输入大小是416x416，那么骨架网络最后层输出shape为(b,1024,13,13)，也就是说采用固定stride=32。如果直接在骨架网络最后一层再接几个卷积层，然后进行预测完全没有问题。作者认为13x13的特征图对于大物体检测来说是够了，但是对于小物体，特征图提供的信息就不一定够了，很可能特征已经消失了，故作者结合特征融合思想，提出一种新的passthrough层来产生更精细的特征图，本质上是一种特征聚合操作，目的是增强小物体特征图信息。

<img src="/assets/objectdetection/img/1/YOLOv2-1.png"  />

上面的第一个输入箭头是darknet-19的最后一个max pool运行前的特征图，假设图片输入是416x416，那么最后一个max  pool层输入尺度为26x26x512，在该层卷积输出后引入一个新的分支：passthrough层，将原来尺度为26x26x512特征图拆分13x13x2048的特征图，然后和darkent-19骨架最后一层卷积的13x13x1024特征图进行concat，得到13x13x3072的特征图，然后经过conv+bn+leakyrelu，得到最终的预测图。上面的图通道画的不对，实际上由于通道太大了，消耗内存太多，故作者实际上做法是先将26x26x512降维为26x26x64，然后拆分为13x13x256,concat后，得到13x13x1280的特征图。

### 正负样本定义

**(1) 正负属性定义** 
 在设定anchor后就需要对每个anchor位置样本定义正负属性了。其规则和yolov1一样简单：**保证每个gt bbox一定有一个唯一的anchor进行对应，匹配规则就是IOU最大**。具体就是：对于某个gt bbox，首先要确定其中心点要落在哪个网格内，然后计算这个网格的5个anchor与该gt  bbox的IOU值，计算IOU值时不考虑坐标，只考虑形状(因为此处anchor没有坐标xy信息)，所以先将anchor与gt  bbox的中心点对齐(简单做法就是把anchor的xy设置为gt  bbox的中心点即可)，然后计算出对应的IOU值，IOU值最大的那个anchor与gt bbox匹配，对应的预测框用来预测这个gt  bbox。其余anchor暂时算作负样本。

有1个情况需要思考清楚：假设有2个gt  bbox的中心都落在同一个网格，且形状差异较大，此时这2个gt  bbox应该会匹配到不同的anchor，这是我们希望的。但是如果差异比较小，导致都匹配上同一个anchor了，那么后一个gt  bbox会把前一个gt bbox匹配的anchor覆盖掉，导致前面的gt bbox变成负样本了，这就是常说的**标签重写**问题。

上述匹配规则在不同复现版本里面有不同实现(官方代码也有，但是从来没有开启该设置)，基于上述匹配规则并且借鉴ssd的做法，可以**额外加上一个匹配规则**：当每个gt bbox和最大iou的anchor匹配完成后，对剩下的anchor再次和对应网格内的gt  bbox进行匹配(必须限制在对应网格内，否则xy预测范围变了)，当该anchor和gt  bbox的iou大于一定阈值例如0.7后，也算作正样本，即该anchor也负责预测gt bbox。这样的结果就是一个gt  bbox可能和好几个anchor匹配，增加了正样本数，理论上会更好。

**(2) 忽略属性定义** 
 此时已经确定了所有anchor所对应的正负属性了，但是我们可以试图分析一种情况：假设某个anchor的预测bbox和某个gt  bbox的iou为0.8，但是在前面的匹配策略中认为其是负样本，这是非常可能出现的，因为每个gt  bbox仅仅和一个anchor匹配上，对于其附近的其余anchor强制认为是负样本。此时由于该负样本anchor预测的情况非常好，如果强行当做负样本进行训练，给人感觉就是不太对劲，但是也不能当做正样本(因为匹配规则就是如此设计的)，此时我们可以把他**当做灰色地带的样本也就是忽略样本，对这类anchor的预测值不计算loss即可，让他自生自灭吧！**

故作者新增了**忽略样本**这个属性(算是yolo系列正负样本定义的一个特色吧)，具体计算过程是：遍历每个anchor的预测值，如果anchor预测值和其余所有gt bbox的所有iou值中，只要有一个iou值大于阈值(通常是0.6)，则该anchor预测值忽略，不计算loss，**对中间灰色区域就不管了，保证学习的连续性。**

**归纳：**

1. 遍历每个gt bbox，首先判断其中心点落在哪个网格，然后和那个网格内的所有anchor计算iou，iou最大的anchor就负责匹配即正样本
2. 考虑额外规则：当每个gt bbox和最大iou的anchor匹配完成后，对剩下的anchor再次和对应网格内的gt  bbox进行匹配(必须限制在对应网格内，否则xy预测范围变了)，当该anchor和gt  bbox的iou大于一定阈值例如0.7后，也算作正样本，即该anchor也负责预测gt bbox
3. 遍历每个负anchor的预测值，如果anchor预测值和其余所有gt bbox的所有iou值中(不需要网格限制)，只要有一个iou值大于阈值，则该anchor预测值忽略，不计算loss

### bbox编解码

<img src="/assets/objectdetection/img/1/yolo-head.png" style="zoom:50%;" />

yolov2的bbox编解码结合了yolov1和ssd的做法，具体是：对于xy中心点值的预测不变，依然是预测相对当前网格左上角偏移。但是对于wh的预测就不同了，主要原因是有anchor了，其wh预测值是gt bbox的宽高除以anchor的wh，然后取log操作即可(和ssd的bbox编解码是一样的)，并且gt  box和anchor尺度都会先映射到特征图上面再计算。可以发现**xy的预测范围是0~1，但是wh的预测范围不定**。

确定了编码过程，那么解码过程也非常简单，就是上面图中的公式，cx,cy的范围是0～13，因为xy预测范围是0~1，故对预测的tx,ty会先进行sigmoid操作(**yolov1没有进行sigmoid**)，然后加上当前网格左上角坐标，最后乘上stride即可得到bbox的中心坐标，而wh就是预测值tw,th进行指数映射然后乘上特征图尺度的anchor宽高值，最后也是乘上stride即可。

相比于yolov1的编解码方式，采用anchor有诸多好处，最主要是可以克服yolov1早期训练极其不稳定问题，原因是其在早期训练时候每个网格都会输出B个任意形状的预测框，这种任意输出可能对某些特定大小的物体有好处，但是会阻碍其他物体学习，导致梯度出现瞬变，不利于收敛。而通过引入基于统计物体wh分布的先验anchor，**相当于限制了回归范围，训练自由度减少了，自然收敛更加容易，训练更加稳定了。**

### loss设计

<img src="/assets/objectdetection/img/1/YOLOv2-2.png"  />

和yolov1一样，所有分支loss都是l2 loss，同时为了平衡正负样本，其也有设置不同分支的loss权重值。

**(1) bbox回归分支loss**

首先$1k^{truth}$的含义是所有正样本为1，否则是0，也就是说bbox回归分支仅仅计算正样本loss。$truth^r$是gt bbox的编码target值，$b^r$是回归分支预测值tx,ty,tw,th，对这4个值计算l2 loss即可。

1t < 128000是指前128000次迭代时候额外考虑一个回归loss，其中$prior^r$表示anchor，表示预测框与先验框anchor的误差，注意不是与gt bbox的误差，可能是为了在训练早期使模型更快学会先预测先验框的位置。 **这个loss的目的是希望忽略xy的学习，而侧重于将预测wh学习出anchor的形状，可能有助于后面的收敛**。

**(2) 分类回归loss** 
 分类分支预测值是$b^c ，truth^c$表示对应的one-hot类别编码，该分支也仅仅是计算正样本anchor。

**(3) 置信度分支loss** 
 对于置信度分支loss计算和yolov1里面完全相同，对于正样本其label是该anchor预测框结果解码后还原得到预测bbox，然后和gt  bbox计算iou，该iou值作为label，用于表示有没有物体且当有物体时候预测bbox的准确性，对于负样本其label=0。上图中$1_{maxIOU}<Thresh$的其实就是表示负样本，可以看出忽略样本全程都是不参与loss计算的。

### 推理流程

对于每个预测框，首先根据类别置信度确定其类别与分类预测分值，将类别概率和confidence值相乘。然后根据阈值（如0.5）过滤掉阈值较低的预测框。对于留下的预测框进行解码，根据先验框得到其真实的位置参数。解码之后，一般需要根据置信度进行降序排列，然后仅保留top-k个预测框。最后就是进行NMS算法，过滤掉那些重叠度较大的预测框，最后剩余的预测框就是检测结果了。

### 总结

通过分析yolov1的召回率低、定位精度差的缺点，并且结合ssd的anchor策略，提出了新的yolov2算法，在BN、新网络darknet19、kmean自动anchor聚类、全新bbox编解码、高分辨率分类器微调、多尺度训练和passthrough的共同作用下，将yolov1算法性能进行了大幅提升，同时保持了yolov1的高速性能。

## YOLOv3

YOLOv3总体思想和YOLOv2没有任何区别，只不过引入了最新提升性能的组件。

**主要贡献**：基于retinanet算法引入了FPN和多尺度预测；基于主流resnet残差设计思想提出了新的骨架网络darknet53 ；不再将所有loss都认为是回归问题，而是分为分类和回归loss，更加符合主流设计思想。

### 网络结构

![](/assets/objectdetection/img/1/yolov3-1.png)

**Yolov3的三个基本组件**：

1. **CBL：**Yolov3网络结构中的最小组件，由**Conv+Bn+Leaky_relu**激活函数三者组成。
2. **Res unit：**借鉴**Resnet**网络中的残差结构，让网络可以构建的更深。
3. **ResX：**由一个**CBL**和**X**个残差组件构成，是Yolov3中的大组件。每个Res模块前面的CBL都起到下采样的作用，因此经过5次Res模块后，得到的特征图是**608->304->152->76->38->19大小**。

**其他基础操作：**

1. **Concat：**张量拼接，会扩充两个张量的维度，例如26x26x256和26x26x512两个张量拼接，结果是26x26x768。Concat和cfg文件中的route功能一样。
2. **add：**张量相加，张量直接相加，不会扩充维度，例如104x104x128和104x104x128相加，结果还是104x104x128。add和cfg文件中的shortcut功能一样。

**Backbone中卷积层的数量：**

每个ResX中包含1+2*X个卷积层，因此整个主干网络Backbone中一共包含**1+（1+2\*1）+（1+2\*2）+（1+2\*8）+（1+2\*8）+（1+2\*4）=52**，再加上一个FC全连接层，即可以组成一个**Darknet53分类网络**。不过在目标检测Yolov3中，去掉FC层，不过为了方便称呼，仍然把**Yolov3**的主干网络叫做**Darknet53结构**。

### 正负样本定义

规则和yolov2完全相同，只不过任何一个gt bbox和anchor计算iou的时候是会考虑三个预测层的anchor，而不是将gt  bbox和每个预测层单独匹配。假设某个gt bbox和第二个输出层的某个anchor是最大iou，那么就仅仅该anchor负责对应gt  bbox，其余所有anchor都是负样本。同样的yolov3也需要基于预测值计算忽略样本。

从yolov2的单尺度预测变成了yolov3多尺度预测，其匹配规则为：

1. 遍历每个gt  bbox，首先判断其中心点落在哪3个网格(三个输出层上都存在)，然后和那3个网格内的所有anchor(一共9个anchor)计算iou，iou最大的anchor就负责匹配即正样本，其余8个anchor都暂时是负样本。可以看出其不允许gt bbox在多个输出层上都预测
2. 考虑额外规则：当每个gt bbox和最大iou的anchor匹配完成后，对剩下的anchor再次和对应网格内的gt  bbox进行匹配(必须限制在对应网格内，否则xy预测范围变了)，当该anchor和gt  bbox的iou大于一定阈值例如0.7后，也算作正样本，即该anchor也负责预测gt bbox
3. 遍历每个负anchor的预测值，如果anchor预测值和其余所有gt bbox的所有iou值中(不需要网格限制)，只要有一个iou值大于阈值，则该anchor预测值忽略，不计算loss

### bbox编解码

规则和yolov2完全相同，正样本的xy预测值是相对当前网格左上角的偏移，而wh预测值是gt bbox的wh除以anchor的wh(注意wh是在特征图尺度算还是原图尺度算是一样的，解码还原时候注意下就行)，然后取log得到。

### loss设计

和YOLOv2不同的是，对于分类和置信度预测值，其采用的不是l2 loss，而是bce loss，而回归分支依然是l2 loss。

分类问题一般都是用ce或者bce loss，而在retinanet里面提到采用bce  loss进行多分类可以避免类间竞争，对于coco这种数据集是很有好处的；其次在机器学习中知道对于逻辑回归问题采用bce  loss是一个凸优化问题，理论上优化速度比l2 loss快；而且分类输出就是一个概率分布，采用分类常用loss是最常规做法，没必要统一成回归问题。

### 推理流程

1. 遍历每个输出层，对xy预测值采用sigmoid变成0~1,然后进行解码，具体是对遍历每个网格，xy预测值加上当前网格坐标，然后乘上stride即可得到预测bbox中心坐标，对于wh预测值先进行指数计算，然后直接乘上anchor的wh即可，此时就可以还原得到最终的bbox
2. 对置信度分支采用sigmoid变成0~1，分类分支由于是bce loss故也需要采用sigmoid操作
3. 利用置信度预测将置信度值低于预测的预测值过滤掉
4. 如果剩下的预测bbox数目多于设置的nms前阈值个数1000，则直接对置信度值进行从大到小topk排序，提取前1000个，三个输出层最多构成3000个预测bbox
5. 对所有预测框进行nms即可

### 总结

yolov3可以认为是当前目标检测算法思想的集大成者，其通过引入主流的残差设计、FPN和多尺度预测，将one-stage目标检测算法推到了一个速度和精度平衡的新高度。由于其高速高精度的特性，在实际应用中通常都是首选算法。

## YOLOv4

YOLOv4算是当前各种trick的集大成者，尝试了许多提升性能的组件。

### 网络结构

![](/assets/objectdetection/img/1/yolov4-1.png)

**YOLOv4的五个基本组件**：

1. **CBM：**YOLOv4网络结构中的最小组件，由Conv+Bn+Mish激活函数三者组成。
2. **CBL：**由Conv+Bn+Leaky_relu激活函数三者组成。
3. **Res unit：**借鉴Resnet网络中的残差结构，让网络可以构建的更深。
4. **CSPX：**借鉴CSPNet网络结构，由卷积层和X个Res unint模块Concate组成。
5. **SPP：**采用1×1，5×5，9×9，13×13的最大池化的方式，进行多尺度融合。

**其他基础操作：**

1. **Concat：**张量拼接，维度会扩充，和YOLOv3中的解释一样，对应于cfg文件中的route操作。
2. **add：**张量相加，不会扩充维度，对应于cfg文件中的shortcut操作。

**Backbone中卷积层的数量：**

和YOLOv3一样，再来数一下Backbone里面的卷积层数量。

每个CSPX中包含5+2xX个卷积层，因此整个主干网络Backbone中一共包含1+（5+2x1）+（5+2x2）+（5+2x8）+（5+2x8）+（5+2*4）=72。

**YoloV4的创新之处:**

1. **输入端：**这里指的创新主要是训练时对输入端的改进，主要包括**Mosaic数据增强、cmBN、SAT自对抗训练**
2. **BackBone主干网络：**将各种新的方式结合起来，包括：**CSPDarknet53、Mish激活函数、Dropblock**
3. **Neck：**目标检测网络在BackBone和最后的输出层之间往往会插入一些层，比如YOLOv4中的**SPP模块**、**FPN+PAN结构**
4. **Prediction：**输出层的锚框机制和Yolov3相同，主要改进的是训练时的损失函数**CIOU_Loss**，以及预测框筛选的nms变为**DIOU_nms**

## YOLOv5

### 网络结构

![](/assets/objectdetection/img/1/yolov5s_1.png)

**（1）输入端：**Mosaic数据增强、自适应锚框计算、自适应图片缩放
**（2）Backbone：**Focus结构，CSP结构应用于**Neck**中
**（3）Neck：**FPN+PAN结构
**（4）Prediction：**GIOU_Loss

### 正负样本定义

yolov5匹配规则为：

(1) 对于任何一个输出层，抛弃了基于max  iou匹配的规则，而是直接采用shape规则匹配，也就是该bbox和当前层的anchor计算宽高比，如果宽高比例大于设定阈值，则说明该bbox和anchor匹配度不够，将该bbox过滤暂时丢掉，在该层预测中认为是背景 。
(2) 对于剩下的bbox，计算其落在哪个网格内，同时利用四舍五入规则，找出最近的两个网格，将这三个网格都认为是负责预测该bbox的，可以发现粗略估计正样本数相比前YOLO系列，至少增加了三倍。

<img src="/assets/objectdetection/img/1/yolov5_1.jpeg"  />

## bbox编解码说明

**Yolov3边框预测**

<img src="/assets/objectdetection/img/1/yolo-head.png" style="zoom:50%;" />

$C_x、C_y$是feature map中grid cell的左上角坐标，每个grid cell在feature map中的宽和高均为1。图中的情形时，这个bbox边界框的中心属于第二行第二列的grid cell，它的左上角坐标为(1,1)，故$C_x=1,C_y=1$。公式中$P_w、P_h$为预设的anchor box映射到feature map中的宽和高(anchor box原本设定是相对于416\*416坐标系下的坐标，除以stride如32映射到feature map坐标系中)。得到边框坐标值是$b_x,b_y,b_w,b_h$即边界框bbox相对于feature map的位置和大小。但网络实际上的学习目标是$t_x,t_y,t_w,t_h$这４个offsets，其中$t_x,t_y$是预测的坐标相对于$C_x、C_y$的偏移值，$t_w,t_h$是相对于$P_w、P_h$的尺度缩放，有了这４个offsets，自然可以根据之前的公式去求得真正需要的$b_x,b_y,b_w,b_h$４个坐标。通过学习偏移量，就可以通过网络原始给定的anchor box坐标经过线性回归微调（平移加尺度缩放）去逐渐靠近groundtruth。

这里需要注意的是，虽然输入尺寸是416x416，但原图是按照纵横比例缩放至416x416的， **取 min(w/img_w, h/img_h)这个比例来缩放，保证长的边缩放为需要的输入尺寸416，而短边按比例缩放不会扭曲**，img_w,img_h是原图尺寸768,576, 缩放后的尺寸为new_w, new_h=416,312，需要的输入尺寸是w,h=416,416。剩下的灰色区域用(128,128,128)填充即可构造为416x416。不管训练还是测试时都需要这样操作原图。而且我们注意**yolov3需要的训练数据的label是根据原图尺寸归一化了的，这样做是因为怕大的边框的影响比小的边框影响大**，因此做了归一化的操作，这样大的和小的边框都会被同等看待了，而且训练也容易收敛。既然label是根据原图的尺寸归一化了的，自己制作数据集时也需要归一化才行。

<img src="/assets/objectdetection/img/1/yolo-head-2.png" style="zoom: 50%;" />

这里解释一下anchor box，YOLOv3为每种FPN预测特征图（13x13,26x26,52x52）设定3种anchor box，总共聚类出9种尺寸的anchor box。在COCO数据集这9个anchor  box是：(10x13)，(16x30)，(33x23)，(30x61)，(62x45)，(59x119)，(116x90)，(156x198)，(373x326)。分配上，在最小的13x13特征图上由于其感受野最大故应用最大的anchor box  (116x90)，(156x198)，(373x326)，（这几个坐标是针对416x416下的，当然要除以32把尺度缩放到13x13），适合检测较大的目标。中等的26x26特征图上由于其具有中等感受野故应用中等的anchor box  (30x61)，(62x45)，(59x119)，适合检测中等大小的目标。较大的52x52特征图上由于其具有较小的感受野故应用最小的anchor  box(10x13)，(16x30)，(33x23)，适合检测较小的目标。特征图的每个像素（即每个grid）都会有对应的三个anchor box，如13x13特征图的每个grid都有三个anchor box  (116x90)，(156x198)，(373x326)（这几个坐标需除以32缩放尺寸）。

那么4个坐标$t_x,t_y,t_w,t_h$是怎么求出来的呢？

$C_x,C_y,P_w,P_h$是预设的anchor box在feature map上目标中心点grid cell位置和宽高。

$G_x,G_y,G_w,G_h$是ground truth除以stride映射到feature map的4个坐标。

YOLOv3里是$G_x,G_y$减去grid cell左上角坐标$C_x,C_y$，为需要预测的偏移。


$$
t_x=G_x-C_x\\
t_y=G_y-C_y\\
t_w=log(G_w/P_w)\\
t_h=log(G_h/P_h)\\
$$
$$
b_x=\sigma(t_x)+C_x\\
b_y=\sigma(t_y)+C_y\\
b_w=P_we^{t_w}\\
b_h=P_he^{t_h}\\
$$

不直接回归bounding box的长宽而是**尺度缩放到对数空间，是怕训练会带来不稳定的梯度。**因为如果不做变换，直接预测相对形变$t_w,t_h$，那么要求$t_w,t_h>0$，因为框的宽高不可能是负数。这样，是在做一个有不等式条件约束的优化问题，没法直接用SGD来做。所以先取一个对数变换，将其不等式约束去掉。

边框回归最简单的想法就是通过平移加尺度缩放进行微调。边框回归为何只能微调？当输入的 Proposal 与 Ground Truth 相差较小时，即IOU很大时(RCNN 设置的是 IoU>0.6)， 可以认为这种变换是一种**线性变换**， 那么**我们就可以用线性回归**（线性回归就是给定输入的特征向量 X, 学习一组参数 W, 使得经过线性回归后的值跟真实值 Y(Ground Truth)非常接近. 即Y≈WX ）**来建模对窗口进行微调**， 否则会导致训练的回归模型不work（当 Proposal跟 GT 离得较远，就是复杂的非线性问题了，此时用线性回归建模显然就不合理了）。

**Yolov5边框预测**

<img src="/assets/objectdetection/img/1/yolov5_1.jpeg"  />

yolov5为了增加正样本数量增加了最近两个网格进行预测，编码解码略有不同


$$
b_x=\sigma(t_x)*2-0.5+C_x\\
b_y=\sigma(t_y)*2-0.5+C_y\\
b_w=P_w*(t_w*2)^2\\
b_h=P_h*(t_h*2)^2\\
$$



yolov3用BCE计算所有正例负例conf loss，用CE计算正例的cls loss，用BCE计算xy loss，用MSE计算wh loss

yolov4用CIou计算bbox loss

yolov5的bbox分支用的是GIou loss
