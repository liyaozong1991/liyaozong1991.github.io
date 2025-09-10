---
categories: [机器学习]     # TAG names should always be lowercase
math: true
title: BCEWithLogloss公式推导
---

## 1. 引言
直接使用这个函数，解决了rank模型中，部分稀疏目标无法正常训练的问题，学习一下其推导过程。
## 2. 二分类问题

一般我们在模型训练中，会使用sigmoid函数作为激活函数，将模型的输出映射到(0,1)之间，作为二分类问题的预测概率。 
但是，sigmoid函数有一些问题，比如：
- 当输入非常大或非常小时，sigmoid函数的输出接近0或1，而梯度接近0，这会导致梯度消失问题。
- 当输入为0时，sigmoid函数的输出为0.5，而梯度为0.25，这会导致模型训练不稳定。

为了解决这个问题，PyTorch提供了BCEWithLogitsLoss函数，它将sigmoid函数和二分类交叉熵损失函数合并在一起，避免了梯度消失问题。

sigmoid函数的公式为：

$$
\begin{equation}
\begin{split}
\sigma(x) = \frac{1}{1 + e^{-x}}
\end{split}
\end{equation}
$$

二分类问题的损失函数为：

$$
\begin{equation}
\begin{split}
L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]
\end{split}
\end{equation}
$$

其中，$y$ 是真实标签，$\hat{y}$ 是模型的预测概率。

## 3. BCEWithLogitsLoss推导

将上面的两个公式结合一下，得到BCEWithLogitsLoss的公式为：
$$ 
\begin{equation}
\begin{split}
L(y, \hat{y}) = -[y \log(\sigma(\hat{y})) + (1 - y) \log(1 - \sigma(\hat{y}))] 
\end{split}
\end{equation}
$$

其中，$\sigma(\hat{y})$ 是模型的预测概率，$y$ 是真实标签。

然后再把sigmoid函数展开，得到：

$$ 
\begin{equation}
\begin{split}
L(y, x) &= -[y \log(\frac{1}{1 + e^{-x}}) + (1 - y) \log(1 - \frac{1}{1 + e^{-x}})]  \\
&= -[-y \log(1 + e^{-x}) + (1 - y) \log(\frac{e^{-x}}{1 + e^{-x}})]  \\
&= -[-y \log(1 + e^{-x}) + (1 - y) (-x - \log(1 + e^{-x}))]  \\ 
&= y \log(1 + e^{-x}) + (1 - y) (x + \log(1 + e^{-x}))  \\  
&= y \log(1 + e^{-x}) + x + \log({1 + e^{-x}}) -xy -y \log(1+e^{-x}) \\
&= \log(1 + e^{-x}) + x - xy
\end{split}
\end{equation}
$$

其中x是模型没有经过sigmoid函数的输出，y是真实标签。

上面的公式，当x>=0时，可以表示为：

$$
\begin{equation}
\begin{split}
L(y, x) &= \log(1 + e^{-|x|}) + x - xy
\end{split}
\end{equation}
$$

当x<0时

$$
\begin{equation}
\begin{split}
L(y, x) &= \log(1 + e^{-x}) + x - xy \\
&= \log(\frac{e^{x}+1}{e^{x}}) + x - xy \\
&= \log(1 + e^{x}) - x + x -xy \\ 
&= \log(1+e^{x}) - xy \\
&= \log(1+e^{-|x|}) - xy
\end{split}
\end{equation}
$$

最终，我们将两种情况合并，得到公式：

$$
\begin{equation}
\begin{split}
L(y, x) &= max(x,0) + \log(1 + e^{-|x|}) - xy
\end{split}
\end{equation}
$$

公式4和公式7是等价的，我们之间从公式4求导可以比较方便的得出损失函数的导数：

$$
\begin{equation}
\begin{split}
\frac{\partial L(y, x)}{\partial x} &= \frac{\partial (\log(1 + e^{-x}) +x - xy)}{\partial x} \\
&= \frac{1}{1 + e^{-x}} - y \\
&= \sigma(x) - y
\end{split}
\end{equation}
$$

那么把损失函数写成公式7的形式，保证计算loss的稳定性，因为无论x是正数还是负数，$ -|x| <0 $, 保证了指数运算结果的范围在(0, 1]，不会越界。