Restricted Boltzmann Machine (RBM)
==================================

Overview
--------
Restricted Boltzmann Machine (RBM) is a type of energy-based, generative, stochastic artificial nerual network. 
It is energy-based beacuse it associates a scalar value to each state of the model. The learning then can be defined 
as finding an energy function that assigned the desired configurations a low-energy value. 
RBM is generative beacuse it learns a probability distribution that fits the training data as well as possible.
It is a stochastic network beacuse its units use a stochastic activation function, which work by introducing noise to the data.
RBM was introduced by Paul Smolensky in 1986, but was later become widely used after Geoffrey Hinton and his team 
introduced sucsseful applications of RBM.  


Examples
--------

There two Jupyter Notebooks that explain how to train RBMs using xRBM:



API
---
:ref:`rbm_label`



References
----------

Smolensky, Paul (1986). "Chapter 6: Information Processing in Dynamical Systems: Foundations of Harmony Theory" (PDF). In Rumelhart, David E.; McLelland, James L. Parallel Distributed Processing: Explorations in the Microstructure of Cognition, Volume 1: Foundations. MIT Press. pp. 194–281. ISBN 0-262-68053-X. `PDF <http://stanford.edu/~jlmcc/papers/PDP/Volume%201/Chap6_PDP86.pdf>`_

Training Products of Experts by Minimizing Contrastive Divergence Geoffrey E. Hinton Neural Computation 2002 14:8, 1771-1800 `DOI <https://dx.doi.org/10.1162/089976602760128018>`_, `PDF <http://www.cs.toronto.edu/~fritz/absps/tr00-004.pdf>`_