# Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

## Abstract: 
This post provides an overview of a phenomenon called "Super Convergence" where we can train a deep neural network in order of magnitude faster compared to conventional training methods. One of the key elements is training the network using "One-cycle policy" with maximum possible learning rate.

*An insight that allows "Super Convergence" in training is the use of large learning rates that regularizes the network, hence requiring a reduction of all other forms of regularization in order to preserve an optimal balance between underfitting and overfitting.*


## Motivation:
You might be wondering that training a model to 94% (high) test accuracy on CIFAR10 in about 75 epochs is a meaningless exercise since state-of-the-art is already above 98%. But don't you think, "State of the art" accuracy is an ill-conditioned target in the sense that throwing a larger model, more hyperparameter tuning, more data augmentation or longer training time at the problem will typically lead to accuracy gains, making a fair comparison between different works a delicate task. Moreover, the presence of "Super Convergence" is relevant to understanding why deep networks generalize so well. The plot below illustrates the "Super Convergence" on the CIFAR10 dataset. We can easily observe that with the modified learning rate schedule we achieve a higher final test accuracy (92.1%) than with typical training (91.2%) and that too, only after a few iterations.

![1](https://user-images.githubusercontent.com/41862477/49628809-66707e00-fa0c-11e8-9045-822c62582faa.JPG) 

## Super-convergence
So let's come to the point quickly and discuss how can we achieve these state of the art results in far lesser number of training iterations. Many people still hold an opinion that training a deep neural network with the optimal hyperparameters is black magic because there are just so many hyper-parameters that one needs to tune. What kind of learning rate policy to follow, what kernel size to pick for the architecture, what weight decay and dropout value will be optimal for the regularization? So, let's break this stereotype and try to unleash some of these black arts.

*We will start with LR Range test that helps you find the maximum Learning rate that you can use to train your model (most important hyper-parameter). Then we will try to run Grid Search CV for the remaining parameters (weight decay and dropout) to find the best value for them.*


### One-Cycle Policy
To achieve super-convergence, we will use "One-Cycle" Learning Rate Policy which requires specifying minimum and maximum learning rate. The Lr Range test gives the maximum learning rate, and the minimum learning rate is typically 1/10th or 1/20th of the max value. One cycle consists of two step sizes, one in which Lr increases from the min to max and the other in which it decreases from max to min. In our case, one cycle will be a bit smaller than the total number of iterations/epochs and in the remaining iterations, we will allow the learning rate to decrease several orders of magnitude lesser than its initial value. The following plot illustrates the One-cycle policy better, left one shows the variation of the cyclical Learning rate and the right one shows the same for the cyclical Momentum.

![2](https://user-images.githubusercontent.com/41862477/49628810-66707e00-fa0c-11e8-8595-25851c8997b8.JPG)

*The motivation for the "One Cycle" policy was the following: The learning rate initially starts small to allow convergence to begin but as the network traverses the flat valley, the learning rate is large, allowing for faster progress through the valley. In the final stages of the training, when the training needs to settle into the local minimum, the learning rate is once again reduced to a small value.*

![3](https://user-images.githubusercontent.com/41862477/49628811-66707e00-fa0c-11e8-8d67-132366dfea61.JPG)

### Learning_Rate Finder
This technique to find max learning rate was first introduced by a great researcher Leslie Smith in his paper which goes into more detail about the benefits of the use of Cyclical learning rate and Cyclical momentum. We start the pre-training with a pretty small learning rate and then increase it linearly (or exponentially) throughout the run. This provides an overview of how well we can train the network over a range of learning rate. With a small learning rate, the network begins to converge and, as the learning rate increases, it eventually becomes too large and causes the test accuracy/loss to diverge suddenly. Typical curves would look similar to the one attached below. The second plot illustrates the independence between the number of training iterations and the accuracy achieved.

![4](https://user-images.githubusercontent.com/41862477/49628812-67091480-fa0c-11e8-9455-c74432bc0a59.JPG)
![5](https://user-images.githubusercontent.com/41862477/49628813-67091480-fa0c-11e8-9667-35e5763be8a5.JPG)

#### *Why does a large Learning rate act like a regularizer?
The LR Range test shows evidence of regularization through results which shows an increasing training loss and decreasing test loss while the learning rate increases from approximately 0.2 to 2.0 when training with the Cifar-10 dataset and a Resnet-56 architecture, which implies that regularization is happening while training with these large learning rates. 
Moreover, the definition says regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error. So, we can now infer that a large learning rate works like a regularizer and helps us achieve an optimal balance between overfitting and underfitting.


### Batch Size
As we all know that small batch size induces regularization effects and others have also shown an optimal batch size on the order of 80 for Cifar-10, but contrary to previous work, the paper suggests using a larger batch size when using the "One-Cycle" policy. The batch capacity should only be limited because of memory constraints, not by anything else since larger batch sizes enable us to use larger learning rates. Although the benefits of larger batch sizes also taper off after some point and 512 seems to be a good choice. The left plot shows the effect of batch size on test accuracy while the right one on test loss.

![6](https://user-images.githubusercontent.com/41862477/49628814-67091480-fa0c-11e8-9385-a05b6d25293a.JPG)

Here, we can observe that batch size of 1024 achieves the best test accuracy in the least number of training iterations compared to others. It is also interesting to contrast the test loss to the test accuracy. *Although larger batch sizes attain lower loss values early in the training, final loss values are least only for the smaller batch sizes, which is the complete opposite to that of accuracy results.*


### Cyclical Momentum
The effect of Momentum and Learning rate on the training dynamics are closely related since the optimal learning rate is dependent on the momentum and momentum is dependent on the learning rate. Momentum is designed to accelerate network training, but its effect on updating the weights is of the same magnitude as the learning rate (can be easily shown for Stochastic Gradient Descent). The optimal training procedure is a combination of an increasing cyclical learning rate and a decreasing cyclical momentum. The max value in the case of cyclical momentum can be chosen after running a grid search for few values (like 0.9, 0.95, 0.97, 0.99) and picking the one which gives the best accuracy. Authors also noticed that final results are nearly independent of the min value of momentum and practically 0.85 works just fine. Following plot shows the effect of momentum on the test accuracy for the CIFAR10 data with ResNet56 architecture.

![7](https://user-images.githubusercontent.com/41862477/49628891-bd765300-fa0c-11e8-914d-0dc3efb92176.JPG)

*From the above plot we can see that decreasing the momentum while increasing the learning rate provides three benefits:*

*(1) A lower test loss, (2) Faster initial convergence, (3) Greater convergence stability over a larger range of learning rate.
One more thing to note that decreasing the momentum and then increasing it is giving much better resulst compared to otherway round.*


### Weight Decay
This one is the last important hyper-paramter that is worth discussing. The amount of regularization must be balanced for each dataset and architecture, and the value of weight decay is a key knob to tune regularization. This requires a grid search over few values to determine the optimal magnitude but usually does not require to search over more than one significant figure.

Using the knowledge of the dataset and architecture we can decide which values to test. For example, a more complex dataset requires less regularization, so testing smaller weight decay values, such as 10−4, 10−5, 10−6, and 0 would suffice. A shallow architecture requires more regularization, so larger weight decay values are tested such as 10−2, 10−3, 10−4. In the grid search we often use values like 3.18e-4, and the reason behind choosing 3 rather than 5 is that a bisection of the exponent is taken into account rather than the bisection of the magnitude itself (i.e., between 10−4 and 10−3 one bisects as 10−3.5 = 3.16 × 10−4).

![8](https://user-images.githubusercontent.com/41862477/49628892-be0ee980-fa0c-11e8-96e3-42fae36254cc.JPG)

From the above plot we can see that weight decay of 1.8e-3 (bisecting the exponent once again b/w -0.5 & -1 i.e. 10^-0.75) allows us to use much larger learning rate, plus giving the minimum test loss compared to other values.

