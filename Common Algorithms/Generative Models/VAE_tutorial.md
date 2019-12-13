#### Variational Auto-Encoder

[Aut-Encoding Variational Bayes](https://arxiv.org/pdf/1312.6114.pdf)

##### 引入
自编码器先通过编码器将原始特征表示映射到隐层特征，再通过解码器重构原始输入，隐层特征和原始输入是一一对应的关系。而变分自编码器是将原始特征映射到一个**概率分布**，解码时从这个概率分布中随机采样生成一个向量作为解码器的输入。这样得到的空间表示是连续、平滑的。

##### 变分下界

在输入$x$时，编码器输出的隐变量$z$是$x$的后验分布，即$p(z | x)$。但是这个真实的后验概率本身不易求，所以用分布$q(z|x)$来近似$p(z | x)$，$q$即是编码器的输出。

数据集$X$的边际分布可通过$x \in X$的边际分布累加得到，即：$\log p_{\theta}(X) = \sum_{x\in X} \log p_{\theta}(x)$，并且对于每个样本$x$都有：

$$
L = \log p(x) = \sum_{z} q(z | x) \log p(x) = \sum_{z} q(z|x) \log \left( \frac{p(z, x)}{p(z | x)} \right) = \sum_{z} q(z | x) \log \left( \frac{p(z, x)}{q(z | x)} \frac{q(z | x)}{p(z | x)} \right) \\
= \sum_{z} q(z | x) \log \left( \frac{p(z, x)}{p(z | x)} \right) + \sum_{z} q(z | x) \log \left( \frac{q(z | x)}{p(z | x)}\right) = L_{B} + KL\left(q(z|x) \Vert p(z | x) \right)
$$

因为第二项KL散度是非负的，所以有$L \geq L_B$，$L_B$被称为变分下界。

又因为$L = p(x)$是固定值，所以如果想最小化$p(z | x)$和$q(z | x)$之间的散度的话，应该最大化$L_B$。

$$
L_B = \sum_{z} q(z | x) \log \left( \frac{p(z, x)}{p(z | x)} \right) = \sum_{z} q(z | x) \log \left( \frac{p(x | z) p(z)}{p(z | x)} \right)\\

= \sum_{z} q(z | x) \log \left( \frac{p(z)}{p(z | x)} \right) + \sum_{z} q(z | x) \log p(x | z) \\
=-KL\left( q(z|x) \Vert p(z) \right) + E_{q(z | x)} \left(\log (p(x | z)) \right) = L_1 + L_2
$$

最大化$L_B$等价于最小化$q(z|x)$和$p(z)$之间的KL散度，同时最大化第二项的期望。因为$q(z|x)$是编码器的输入，并且假设$z$服从高斯分布，所以目标就是让编码器的输出尽可能服从高斯分布。

设编码器的参数为$\phi$，已知$p(z)$服从正态高斯分布，即$p(z) = N(0, I)$，$q(z | x) = N(z; \mu(x, \phi), \delta^2(x, \phi))$，根据KL散度定义有：

$$
L_1 = \int q(z|x) \log \left( \frac{p(z)}{p(z | x)} \right) dz = \int q(z|x) \log p(z) dz - \int q(z|x) p(z|x) dz \\
= \int N(z; \mu, \delta^2) \log N(z; 0, I) dz - \int N(z; \mu, \delta^2) \log N(z; \mu, \delta^2) dz \\
= E_{z \sim N(\mu, \delta^2 )}[\log N(z; 0, I)] -  E_{z\sim N(u, \delta^2)} [\log N(z; \mu, \delta^2)]\\
= E_{z \sim N(\mu, \delta^2)} [\log \left(\frac{1}{\sqrt{2\pi}} \exp (-\frac{z^2}{2}) \right)] - E_{z\sim N(u, \delta^2)} [\log \left( \frac{1}{\sqrt{2\pi \delta^2}} \exp(- \frac{(z - \mu)^2}{2 \delta^2})  \right)]\\
= \left(-\frac{1}{2}\log 2\pi - \frac{1}{2}E_{z \sim N(\mu, \delta^2)}[z^2] \right) - \left( -\frac{1}{2} \log 2\pi - \frac{1}{2}\log \delta^2 - \frac{1}{2\delta^2} E_{z \sim N(u, \delta^2)} [(z - \mu)^2] \right)\\
= \left( -\frac{1}{2}\log 2\pi - \frac{1}{2}(\mu^2 + \delta^2) \right) - \left( -\frac{1}{2} \log 2\pi - \frac{1}{2} (\log \delta^2 + 1) \right) \\
= \frac{1}{2} \left( 1 + \log \delta^ 2 - \mu^2 - \delta^2 \right)
$$


而对于$L_2$，直接求解数似然期望比较复杂，采用MC(Monte Carlo)采样估计：

$$
L_2 = E_{q(z | x)} \left(\log (p(x | z)) \right) \simeq \frac{1}{L} \sum_{l=1}^{L} \log p(x \vert z^{l})
$$
其中$z^l \sim q(z | x)$。不过实际计算中，$L_B$对$\phi$的梯度方差很大，不适用于数值计算，为了稳定性，添加噪音训练：$z^l \sim q(z | x, \epsilon)$，其中$\epsilon$为噪声，$\epsilon \sim p(\epsilon)$。

在minibatch训练的时候，对数似然函数的下界可以通过minibatch来估计：
$$
L_B \simeq \frac{N}{M} \sum_{z \in X_M} (L_1 + L_2) = \frac{N}{M} \sum_{x \in X_M}\left( \frac{1}{2} (1 + \log \delta^2 - \mu^2 -\delta^2) + \frac{1}{L} \sum_{l=1}^{L} \log p(x | z^l) \right) 
$$

可见为了计算$L_B$，使用了两层估计。当$M$较大时，内层估计可由外层估计来完成。实际计算中可取$L=1$。