## 损失函数

$$
L = \sum\limits_{(u,i)\in D_{pos}} softplus(-S_{ui}) + \sum\limits_{(u,j)\in D_{neg}}w_j\cdot softplus(S_{uj} - \log P(j))
$$

其中，正采样可以直接通过 $center \cdot context$ 求和然后softplus计算得到。

商品采样概率根据 $popularity^{0.75}$ 得出，每日更新。

负采样概率则根据局部概率计算，$P(i) = \frac{popularity_i}{\sum\limits_x^n  popularity_x}$

## 优化器

初次使用时使用的Adagrad优化器，尝试在向量数据库中记录优化器参数,但后续发现数据其实有时效性,我们并不希望今日的数据训练后优化器的参数对后续训练有影响,每天都相当于一次重训.
