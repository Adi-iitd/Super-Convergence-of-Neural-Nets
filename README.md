# Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

### Abstract: 
This post describes a phenomenon, which goes by name “super-convergence”, where neural networks can be trained an order of magnitude faster than with standard training methods. One of the key elements of super-convergence is training with one cycle policy and a large maximum learning rate. A primary insight that allows super-convergence training is that large learning rates regularize the training, hence requiring a reduction of all other forms of regularization in order to preserve an optimal regularization balance. 

### Motivation: 
There is a reasonable point of view that says that training a model to 94% test accuracy on CIFAR10 is a meaningless exercise since state-of-the-art is above 98%. "State-of-the-art" accuracy is an ill-conditioned target in the sense that throwing a larger model, more hyperparameter tuning, more data augmentation or longer training time at the problem will typically lead to accuracy gains, making a fair comparison between different works a delicate task. Moreover, the existence of super-convergence is relevant to understanding why deep networks generalize well. 
Fig1 illustrates the super-convergence: 

![1](https://user-images.githubusercontent.com/41862477/49326808-b4553400-f57d-11e8-8931-30121431d806.JPG)

Here we can easily observe that with the modified learning rate schedule, we achieve a higher final test accuracy (92.1%) than typical training (91.2%) after only a few iterations. 

### Super-convergence
So let's come to the point quickly and discuss how can we achieve these state of the art results in far lesser number of training iterations. Many people still hold an opinion that training a deep neural network with the optimal hyperparameters is black magic. There are just so many hyper-parameters that one needs to tune; what kind of learning rate policy to use, what kernel size should we pick in our architecture, what weight decay and dropout value will be optimal to regularize the network optimally?  So, let's break this stereotype and try to unleash some of the black arts. First, we will see how to find the best Learning rate schedule (most important hyper-parameter). The motivation behind such a peculiar policy is the following: The learning rate initially starts small to allow convergence to begin. As the network traverses the flat valley, the learning rate is large, allowing for faster progress through the valley. In the final stages of the training, when the training needs to settle into the local minimum, the learning rate is once again reduced to a small value.

#### One-Cycle Policy
To achieve super-convergence, we will use "One-Cycle" Learning rate Policy and for that, we need to specify minimum and maximum learning rate boundaries. The maximum Learning rate is calculated as described below from the Lr range test and the minimum learning rate is typically 1/10th or 1/20th of the maximum learning rate. One cycle consists of two step sizes, one in which Lr increases from its min value to max and the other in which it decreases from max to its min value. In our case, one cycle will be a bit smaller than the total number of iterations/epochs and in the remaining iterations, we will allow the learning rate to decrease several orders of magnitude less than its initial value. 

#### Learning_Rate Finder
We start the training with a zero or very small learning rate and then increase it in a linear (or exponential) fashion slowly throughout a pre-training run. This provides information on how well the network can be trained over a range of learning rates. With a small learning rate, the network begins to converge and, as the learning rate increases, it eventually becomes too large and causes the test accuracy/loss to diverge suddenly. Typical curves would look like this, the second curve shows the independence between the number of training iterations and the accuracy: 

![2](https://user-images.githubusercontent.com/41862477/49327191-09944400-f584-11e8-8509-ddbde585b8ee.JPG)

#### Why Large learning rate is behaving like a regularizer?
The LR range test reveals evidence of regularization through results which shows an increasing training loss and decreasing test loss while the learning rate increases from approximately 0.2 to 2.0 when training with the Cifar-10 dataset and a Resnet-56 architecture, which implies that regularization is occurring while training with these large learning rates.
Since the definition of regularization suggests “any modification we make to a learning algorithm that is intended to reduce its generalization error”, so large learning rates should be considered as regularizing.

#### Bacth Size
Earlier, small batch sizes have been recommended for regularization effects and others have shown there to be an optimal batch size on the order of 80 for Cifar-10 but contrary to previous work, this one suggests using a larger batch size when using the One-Cycle learning rate schedule. The batch capacity should only be limited due to memory constraints, not by anything else since larger batch sizes enables to use larger learning rates. Although, the benefits of larger batch sizes tapers off after a some point and 512 seems to be a good choice in most cases.

#### Cyclical Momentum
The effect of Momentum and Learning rate are closely inter-wined since the optimal learning rate is dependent on the momentum and momentum is dependent on the learning rate. Momentum is designed to accelerate network training but its effect on updating the weights is of the same magnitude as the learning rate (can be easily shown for Stochastic Gradient Descent). The optimal training procedure is a combination of an increasing cyclical learning rate, where an initial small learning rate permits convergence to begin, and a decreasing
cyclical momentum, where the decreasing momentum allows the learning rate to become larger in the early to middle parts of training. However, if a constant learning rate is used then a large constant momentum (i.e., 0.9-0.99) will act like a pseudo increasing learning rate and will speed up the training. However, use of too large a value for momentum causes poor training results and are commonly visible in early part of the training. Decreasing the momentum while the learning rate increases provides three benefits: (1) A lower minimum test loss, (2) Faster initial convergence, (3) Greater convergence stability over a larger range of learning rate.

#### Weight Decay

