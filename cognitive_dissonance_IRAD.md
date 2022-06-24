
# Latent Dissonance for Robust Learning

### Brief Project Summary (500 chars)

We aim to track class-based activation statistics for hidden layers of neural networks during training using a memory-efficient quantile approximation algorithm. Using three datasets, we will demonstrate that hidden activation distributions enable efficient and sensitive methods for detecting distribution shift, concept drift, and the existence of discrete subpopulations within classes (enabling networks to detect when classes should be split into child classes to improve training).

### Challenge Problem (500 chars)

Very little is known about the activations of hidden neurons within large neural networks, hence their noteriety as "black box" models. Since they are difficult to interpret and large in number, hidden neurons are underexplored by theories on domain shift, robustness, and adaptive learning despite the immense information they contain. Efficient methods of automatically monitoring hidden activations offer potentially disruptive capabilities in distribution shift and training robustness.

### Objective (250 chars)

We aim to pioneer a new approach to neural network training that is predictated on automatic curation and use of class-based activation statistics within hidden layers to detect and mitigate distribution shift without requiring human intervention.

### How is the challenge addressed today (1200 chars)

Most techniques to improve neural network robustness require extensive metadata (e.g. concept labels) or estimations of the distribution of network inputs (e.g. importance weighting). The former requires onerous labeling and is rarely available for a deployed model in a realistic environment, while the latter is time consuming and often inaccurate for high dimensional input spaces (e.g. images). Techniques that examine the activations of hidden layers track minimal information (e.g. mean and variance) which fail to capture distributional information (e.g. multi-modality, skew, kurtosis, etc.) that are common in realistic data. Furthermore, out-of-distribution detection methods typically require extensive human oversight and entirely rely upon human intervention for mitigating detected problems. Preference for deeper networks and a push towards artificial intelligence on edge devices has rendered human oversight as an unscalable solution to concerns of distribution shift. There is an unmet need for efficient methods to track high resolution distributional information about hidden activations and to use this information to mitigate distribution shift problems in an automatic fashion.

### What is your approach (1200 chars)

We plan to track within-class activations distributions for hidden neurons and to use these statistics to automatically tune the training process of neural networks. We have designed a provably unbaised estimator for each hidden neuron's per-class activation probability density consisting of a list of quantiles for each neuron-class pair. The estimator is memory-efficient since it caches only the estimator itself. We will use this estimator to automatically detect and mitigate three phenomena (domain shift in the input, concept drift, and fused classes) for three models (AlexNet, DPN, and VGGVox) on three datasets (ImageNet, Function Map of the World, and VoxCeleb). For domain shift, similar classes (e.g. "terminal" and "hangar") will be fused into one (e.g. "airport") and the makeup of the new class will be gradually altered. For concept drift, one class (e.g. "terminal") will gradually have more of its members labeled as another (e.g. "airport") during training. In each case, our training process must automatically identify the shift (or class fusion) and intervene (either with a preferential data augmentation plan or a class split) to improve performance on validation data.

### Impact (800 chars)

We believe robust learning methods based on comprehensive activation statistics of hidden neurons are best suited to advance state-of-the-art neural network architectures. Our technique is a stepping stone to allow neural networks to dynamically adapt to distribution shift while learning and to add/remove/modify classes and latent concepts during training. Importantly, this research aims to ultimately obviate the need for human oversight and intervention for distribution shift by equipping networks with the ability to devise and implement robust mitigation strategies during training. We see an opportunity for this IRAD work to produce an open source tools that can foster external collaboration and bolster APL's thought leadership in the space.

### Risks (250 chars)

This IRAD is well grounded in theory and will focus primarily on rapid software prototyping to mitigate schedule risk and produce demonstrable capability.

### Follow-on Work (250 chars)

We see this effort culminating in a open source tool and capability demonstration that serves as the basis for a research proposal to sponsors (DARPA/IARPA).

# Side Thoughts
### Making hidden neuron activations independent

For a basic example, imagine a neural network with input neurons whose activations
form the elements of $\vec{x}$. For domain shift purposes, it is useful to know $P(X=\vec{x})$, and this is typically estimated empirically. However, if $\vec{x}$ is of high dimension, it may be difficult to estimate the similarly high dimensional $P(X=\vec{x})$ since the observed data are equally likely under many, very different distributions. Additionally, attempting to estimate $P(X=\vec{x})$ may be wasteful if a lower dimensional vector can capture the information of $X$ with little loss.

One could imagine performing PCA on any input vector before feeding it to a network. This would allow a truncation of the dimensionality of the input space that discards a minimal portion of the variation in the data. If done well, this guarantees elements of $X$ that are uncorrelated with one another. In this case, the elements of $X$ are not independently distributed, since while $x_{1}$ is uncorrelated with $x_{2}$, $\left(x_{1} - \langle x_{1} \rangle\right)$ may be correlated with $x_{2}$ (e.g. the variance of $x_{1}$ depends on $x_{2}$).

With sufficient data, one could imagine that it would be possible to produce a transformation $H(\vec{x})$ such that
$$
\vec{h} = H(\vec{x})
$$
and where $h_{1}$ and $h_{2}$ are independent of each other. If $x_{1}$ and $x_{2}$ are uncorrelated, and an estimate of the standard deviation of $x_{2}$ as a function of $x_{1}$ exists (called $\sigma_{x_{2}}(x_{1})$), we can let
$$
h_{1} = x_{1} \\
h_{2} = x_{2} / \sigma_{x_{2}}(x_{1})
$$
We could even define in the two element case a transformation matrix $G$ such that
$$
\vec{h} = G\vec{x}
$$
$$
G = \left[ \begin{matrix} 1 & 0 \\ 1 & \frac{1}{\sigma_{x_{2}}(x_{1})} \end{matrix} \right]
$$
A challenge arises with extending this to larger vectors. In particular, standard deviation estimates in this formalism become dependent on an increasing number of variables, and this can quickly reduce the density of data to an unusable level. One possible solution is to rely increasingly on $\sigma_{x_{n}}(x_{1},x_{2},...,x_{n-1})=\sigma_{x_{n}}$, e.g. to assume that the variance of high order prinicpal components are in fact independent already.

We have ignored until now the form of $\sigma_{x_{2}}(x_{1})$. It is difficult to recommend a form that is generally good. One obvious constraint would be to maintain that $\sigma_{x_{2}}(x_{1}) > 0$. One way would be to initialize $\sigma_{x_{2}}(\mu_{x_{1}})=\sigma_{x_{2}}$ and to perform a moving estimate of $\sigma_{x_{2}}(x)$ as $x$ moves along the dimension of $x_{1}$ in either direction. New entries to the running variance are weighted by how far they are from the previous, so if the last $x_{1}$ value committed to the running variance is $x_{1,k}$ and the next one is $x_{1,k+1}$, the old estimate is weighted by $(1 - \gamma * (x_{1,k+1} - x_{1,k}))$ and the new estimate is weighted by $\gamma * (x_{1,k+1} - x_{1,k})$ (note we enforce $x_{1,k+1} > x_{1,k}$ by processing the data in order). Calculating an estimate for $\sigma_{x_2}(x_1)$ in this fashion is of complexity $\mathcal{O}(n)$. Doing the same for the $k$ th element of $x$ (factoring dependence on the earlier elements) is of complexity $\mathcal{O}(n^{k})$. However, this process is assumed to terminate after $k$ is large enough such that there is insufficient data, and this implies $m / b^{k} < \varepsilon$ where $m$ is the number of data points, $b$ is a positive integer bin resolution, and $\varepsilon$ is a small positive integer denoting minimum data needed. Thus, the complexity of this approach in the size of data, $m$, is $\mathcal{O}(n^{\log_{b}(m/\varepsilon)})$.

Assuming this is a feasible approach, let's examine the advantages of producing an encoding of $\vec{x}$, $\vec{h}$, with independently distributed elements. Most immediately, we note
$$
P(X)=P(H)=P(\cup_{k=1}^{n}h_{k}=H_{k})
$$
$$
P(X)=\prod_{k=1}^{n}P(h_{k}=H_{k})
$$
Most importantly, this allows for the definition of probability gradients with respect to elements of $X$.

### Ideas that avoid needing independence in hidden activations

Suppose that hidden activations are uncorrelated but not independent (e.g. post PCA). While not guaranteed to do so, distribution shift or unusual inputs could be detected by examining the number of left-of-mean hidden activations and the number of right-of-mean hidden activations. if all $h$ tracked hidden neurons are decorrelated, then for any input sample the number of left-of-mean hidden activations is binomially distributed. The probability of observing $k$ left-of-mean hidden activations is
$$
P(k)=\frac{1}{2^{h}}\frac{h!}{(h-k)!k!}
$$
