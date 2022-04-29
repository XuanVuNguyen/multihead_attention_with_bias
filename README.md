# Multihead Attention layer that supports attention bias and autoregression
This is the implementation of the Multihead Attention layer from the famous ["Attention is all you need"](https://arxiv.org/abs/1706.03762), except I added a bias term in the logit before the softmax function, as I have seen this term increasingly appears in literature, for example, [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2) and [Graphormer](https://arxiv.org/abs/2106.05234). The calculation of attention weights is as followed:

$$A = softmax(\frac{Q K^\top}{\sqrt{d}}+b)V$$

with $Q$, $K$, and $V$ is the projection of query, key, and value, respectively, $d$ is the dimention of each attention head, and $b$ is the bias term.

This layer also supports auto-regression, as in the decoder of the original Transformer. In details, this layer is designed to be able to store the projections of key and value in the argument `keys_values_cache` in order to reuse these projections for subsequent time-steps when performing auto-regression.

The detail usage of this layer can be found in [demo.ipynb](demo.ipynb).