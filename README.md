# Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

## Abstract: 
This post describes a phenomenon “Super-Convergence” where neural networks can be trained an order of magnitude faster compared to standard training methods. One of the key elements is training the networks using one cycle policy with maximum possible learning rate. A primary insight that allows super-convergence training is that large learning rates regularize the training, hence requiring a reduction of all other forms of regularization in order to preserve an optimal regularization balance.

## Motivation:
You might be wondering that training a model to 94% test accuracy on CIFAR10 is a meaningless exercise since state-of-the-art is already above 98%. "State-of-the-art" accuracy is an ill-conditioned target in the sense that throwing a larger model, more hyperparameter tuning, more data augmentation or longer training time at the problem will typically lead to accuracy gains, making a fair comparison between different works a delicate task. Moreover, the existence of super-convergence is relevant to understanding why deep networks generalize so well. The figure below illustrates the super-convergence on the CIFAR10 dataset. We can also easily observe that with the modified learning rate schedule, we achieve a higher final test accuracy (92.1%) than typical training (91.2%) only after a few iterations. 

![1](https://user-images.githubusercontent.com/41862477/49328753-972f5e00-f59b-11e8-9fbf-16465a08c672.JPG) 

## Super-convergence
So let's come to the meaty part quickly and discuss how can we achieve these state of the art results in far lesser number of training iterations. Many people still hold an opinion that training a deep neural network with the optimal hyperparameters is black magic because there are just so many hyper-parameters that one needs to tune; What kind of learning rate policy will be best, what kernel size to pick for the architecture, what weight decay and dropout value will be optimal to add regularization to the network? So, let's break this stereotype and try to unleash some of these black arts. 

First, we will see how to find the boundaries for Learning rate schedule [most important hyper-parameter!!] i.e. Maximum and Minimum Learning rate. Then we will try to run Grid Search CV for the remaining parameters (weight decay and dropout) to find the best value for them.

### One-Cycle Policy
To achieve super-convergence, we will use "One-Cycle" Learning rate Policy and for that, we need to specify minimum and maximum learning rate. The maximum Learning rate is calculated as described below from the Lr range test and the minimum learning rate is typically 1/10th or 1/20th of the maximum learning rate. One cycle consists of two step sizes, one in which Lr increases from the min value to max and the other in which it decreases from max to its original min value. In our case, one cycle will be a bit smaller than the total number of iterations/epochs and in the remaining iterations, we will allow the learning rate to decrease several orders of magnitude less than its initial value. 

The motivation behind such One-Cycle policy is the following: The learning rate initially starts small to allow convergence to begin. As the network traverses the flat valley, the learning rate is large, allowing for faster progress through the valley. In the final stages of the training, when the training needs to settle into the local minimum, the learning rate is once again reduced to a small value. Following figure illustrates the One-cycle policy better. Left plot shows variation of cyclical Learning rate and right plot for the Cyclical Momemtum.

![4](https://user-images.githubusercontent.com/41862477/49328784-0efd8880-f59c-11e8-94f3-35a69260ce97.JPG)

### Learning_Rate Finder
We start the pre-training with a zero or very small learning rate and then increase it in a linear (or exponential) fashion slowly throughout the run. This provides information on how well the network can be trained over a range of learning rates. With a small learning rate, the network begins to converge and, as the learning rate increases, it eventually becomes too large and causes the test accuracy/loss to diverge suddenly. Typical curves would look similar to the one attched below. The second plot illustrates the independence between the number of training iterations and the accuracy achieved. 

![2](https://user-images.githubusercontent.com/41862477/49327191-09944400-f584-11e8-8509-ddbde585b8ee.JPG)

#### Why large Learning rate acts like a regularizer?
The LR range test reveals evidence of regularization through results which shows an increasing training loss and decreasing test loss while the learning rate increases from approximately 0.2 to 2.0 when training with the Cifar-10 dataset and a Resnet-56 architecture, which implies that regularization is occurring while training with these large learning rates. According to the definition, regularization is “any modification we make to a learning algorithm that is intended to reduce its generalization error”, so from here, we can conclude that large learning rate can also be considered as one of the regularization techniques.

### Bacth Size
Earlier, there used to be a popular belief common among people that small batch size induces regularization effects and others have also shown an optimal batch size on the order of 80 for Cifar-10, but contrary to previous work, this one suggests using a larger batch size when using the One-Cycle policy. The batch capacity should only be limited because of memory constraints, not by anything else since larger batch sizes enables to use larger learning rates. Although, the benefits of larger batch sizes also tapers off after a some point but 512 seems to be a good choice in most cases. Left plot shows the effect of batch size on test accuracy while the right one on test loss.  

![5](https://user-images.githubusercontent.com/41862477/49328690-cb564f00-f59a-11e8-8475-6fa13f5c3f76.JPG)

### Cyclical Momentum
The effect of Momentum and Learning rate are closely inter-wined since the optimal learning rate is dependent on the momentum and momentum is dependent on the learning rate. Momentum is designed to accelerate network training but its effect on updating the weights is of the same magnitude as the learning rate (can be easily shown for Stochastic Gradient Descent). The optimal training procedure is a combination of an increasing cyclical learning rate, where an initial small learning rate permits convergence to begin, and a decreasing
cyclical momentum, where the decreasing momentum allows the learning rate to become larger in the early to middle parts of training. However, if a constant learning rate is used then a large constant momentum (i.e., 0.9-0.99) will act like a pseudo increasing learning rate and will speed up the training. However, use of too large a value for momentum causes poor training results and are commonly visible in early part of the training. Decreasing the momentum while the learning rate increases provides three benefits: (1) A lower minimum test loss, (2) Faster initial convergence, (3) Greater convergence stability over a larger range of learning rate.

#### Weight Decay

