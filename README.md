# Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

## Abstract: 
This post provides an overview of a phenomenon called "Super Convergence" where we can train a deep neural network in order of magnitude faster compared to conventional training methods. One of the key elements is training the network using "One-cycle policy" with maximum possible learning rate.

*An insight that allows "Super Convergence" in training is the use of large learning rates that regularizes the network, hence requiring a reduction of all other forms of regularization to preserve a balance between underfitting and overfitting.*


## Motivation:
You might be wondering that training a model to 94% (high) test accuracy on CIFAR10 in about 75 epochs is a meaningless exercise since state-of-the-art is already above 98%. But don't you think, "State of the art" accuracy is an ill-conditioned target in the sense that throwing a larger model, more hyperparameter tuning, more data augmentation or longer training time at the problem will typically lead to accuracy gains, making a fair comparison between different works a delicate task. Moreover, the presence of Super Convergence is relevant to the understanding of the generalization of deep networks.

![1](https://user-images.githubusercontent.com/41862477/49628809-66707e00-fa0c-11e8-9045-822c62582faa.JPG) 

> The plot shown above illustrates the "Super Convergence" on the CIFAR10 dataset. We can easily observe that with the modified learning rate schedule we achieve a higher final test accuracy (92.1%) than with typical training (91.2%) and that too, only in a few iterations.

## Super-convergence
So let's come to the point quickly and discuss how can we achieve these state of the art results in far lesser number of training iterations. Many people still hold an opinion that training a deep neural network with the optimal hyperparameters is black magic because there are just so many hyper-parameters that one needs to tune. What kind of learning rate policy to follow, what kernel size to pick for the architecture, what weight decay and dropout value will be optimal for the regularization? So, let's break this stereotype and try to unleash some of these black arts.

*We will start with LR Range test that helps you find the maximum Learning rate, which you can use to train your model (most important hyper-parameter). Then we will run Grid Search CV for the remaining parameters (weight decay & dropout) to find their best values.*

### Learning_Rate Finder
It was **Leslie Smith** who first introduced this technique to find max learning in his [paper](https://arxiv.org/pdf/1506.01186.pdf), which goes into much more detail, of about the benefits of the use of Cyclical learning rate and Cyclical momentum. We start the pre-training with a pretty small learning rate and then increase it linearly (or exponentially) throughout the run. This provides an overview of how well we can train the network over a range of learning rate. With a small learning rate, the network begins to converge and, as the learning rate increases, it eventually becomes too large and causes the test accuracy/loss to diverge suddenly.

![5](https://user-images.githubusercontent.com/41862477/49628813-67091480-fa0c-11e8-9667-35e5763be8a5.JPG)
![4](https://user-images.githubusercontent.com/41862477/49628812-67091480-fa0c-11e8-9455-c74432bc0a59.JPG)

> Typical curves would look similar to the one attached above, the second plot illustrates the independence between the number of training iterations and the accuracy achieved.

### One-Cycle Policy
To achieve super-convergence, we will use "One-Cycle" Learning Rate Policy which requires specifying minimum and maximum learning rate. The Lr Range test gives the maximum learning rate, and the minimum learning rate is typically 1/10th or 1/20th of the max value. One cycle consists of two step sizes, one in which Lr increases from the min to max and the other in which it decreases from max to min. In our case, one cycle will be a bit smaller than the total number of iterations/epochs and in the remaining iterations, we will allow the learning rate to decrease several orders of magnitude lesser than its initial value. The following plot illustrates the One-cycle policy better - left one shows the cyclical Learning rate and the right one shows the cyclical Momentum.

![2](https://user-images.githubusercontent.com/41862477/49628810-66707e00-fa0c-11e8-8595-25851c8997b8.JPG)

*The motivation for the "One Cycle" policy was the following: The learning rate initially starts small to allow convergence to begin but as the network traverses the flat valley, the learning rate becomes large, allowing for faster progress through the valley. In the final stages of the training, when the training needs to settle into the local minimum, the learning rate is once again reduced to a small value.*

![3](https://user-images.githubusercontent.com/41862477/49628811-66707e00-fa0c-11e8-8d67-132366dfea61.JPG)

> The left plot shows the visualization of, how training transverses a loss function topology, whereas the right plot shows a close-up of the end of optimization.

***

***Why does a large Learning rate act like a regularizer?***

The **LR Range test** shows evidence of regularization through results, which shows an increasing training loss and decreasing test loss while the learning rate increases from approximately 0.2 to 2.0 when training with the Cifar-10 dataset and a Resnet-56 architecture, which implies that regularization is happening while training with these large learning rates. Moreover, the definition says **regularization** is any modification we make to a learning algorithm that is intended to reduce its generalization error.

***

### Batch Size
As we all know that small batch size induces regularization effects and some have also shown an optimal batch size on the order of 80 for Cifar-10, but contrary to previous work, this paper suggests using a larger batch size when using the **One-Cycle** policy. The batch size should only be limited because of memory constraints, not by anything else since larger batch size enables us to use larger learning rate. Although, the benefits of larger batch sizes taper off after some point. 

![6](https://user-images.githubusercontent.com/41862477/49629231-7ee19800-fa0e-11e8-8b86-d9c5003d9eca.JPG)

> The left plot shows the effect of batch size on test accuracy while the right one on test loss. Here, we can observe that batch size of 1024 achieves the best test accuracy in the least number of training iterations compared to others. 

It is also interesting to contrast the test loss to the test accuracy. *Although larger batch size attains a lower loss value early in the training, the final loss value is least only for the smaller batch size, which is the complete opposite to that of accuracy result.*

***

### Cyclical Momentum
The effect of Momentum and Learning rate on the training dynamics are closely related since both of them are dependent on each other. Momentum is designed to accelerate network training, but its effect on updating the weights is of the same magnitude as the learning rate (can be easily shown for Stochastic Gradient Descent). 

The optimal training procedure is a combination of an increasing cyclical learning rate and a decreasing cyclical momentum. The max value in the case of cyclical momentum can be chosen after running a **grid search** for few values (like 0.9, 0.95, 0.97, 0.99), and picking the one which gives the best test accuracy. Authors also observed that final results are nearly independent of the min value of momentum, and 0.85 works just fine.

![7](https://user-images.githubusercontent.com/41862477/49628891-bd765300-fa0c-11e8-914d-0dc3efb92176.JPG)

> The plot shown above illustrates the effect of momentum on the test accuracy for the CIFAR10 data with ResNet56 architecture.

*Decreasing the momentum while increasing the learning rate provides three benefits:*
- A lower test loss, 
- Faster initial convergence, 
- Greater convergence stability over a larger range of learning rate.

*One more thing to note that decreasing the momentum and then increasing it is giving much better results compared to otherway round.*

***

### Weight Decay
This is the last important hyper-parameter that is worth discussing. The amount of regularization must be balanced for each dataset and architecture, and the value of weight decay is a key knob to tune regularization. This requires a grid search over few values to determine the optimal magnitude but usually does not require to search over more than one significant figure.

Using the knowledge of the dataset and architecture we can decide which values to test. For example, a more complex dataset requires less regularization, so testing smaller weight decay values, such as 10−4, 10−5, 10−6, and 0 would suffice. A shallow architecture requires more regularization, so larger weight decay values are tested such as 10−2, 10−3, 10−4. In the grid search we often use values like 3.18e-4, and the reason behind choosing 3 rather than 5 is that a bisection of the exponent is taken into account rather than the bisection of the magnitude itself (i.e., between 10−4 and 10−3 one bisects as 10−3.5 = 3.16 × 10−4).

![8](https://user-images.githubusercontent.com/41862477/49628892-be0ee980-fa0c-11e8-96e3-42fae36254cc.JPG)

> From the above plot we can see that weight decay of 1.8e-3 (bisecting the exponent once again b/w -0.5 & -1 i.e. 10^-0.75) allows us to use much larger learning rate, plus giving the minimum test loss compared to other values.

*Now following this learning rate schedule with a well-defined procedure to do Grid-Search CV will give you better results with almost a reduction of 50% in the training iterations.*

***

## Result

**The results shown below are for FMNIST dataset trained using techniques discussed above:**

> ***Histogram and Distribution plot of the weights of the first layer (to check **Vanishing gradient** problems)***

![1](https://user-images.githubusercontent.com/41862477/49699358-31278400-fbf6-11e8-9d14-c02ead7397a4.JPG)
![1 1](https://user-images.githubusercontent.com/41862477/49699359-31c01a80-fbf6-11e8-99bd-b431cf6db7da.JPG)

> ***Histogram and Distribution plot of the weights of the last layer***

![2](https://user-images.githubusercontent.com/41862477/49699360-31c01a80-fbf6-11e8-8add-b6a0ea85c8a2.JPG)
![2 1](https://user-images.githubusercontent.com/41862477/49699361-31c01a80-fbf6-11e8-96ad-8cf3b0d0b7ca.JPG)

> ***Architecture Used: WideResBlk Type 1 & WideResBlk Type 2***

![3](https://user-images.githubusercontent.com/41862477/49699362-3258b100-fbf6-11e8-9c47-4c2d66f2a3eb.JPG)
![4](https://user-images.githubusercontent.com/41862477/49699363-3258b100-fbf6-11e8-876f-8916d86b80c7.JPG)

> ***Accuracy***

![5](https://user-images.githubusercontent.com/41862477/49699364-3258b100-fbf6-11e8-880c-0357b3b31fc3.JPG)

> ***LR Range Test***

![6](https://user-images.githubusercontent.com/41862477/49699365-3258b100-fbf6-11e8-8789-f60fba38ed9e.JPG)

> ***GridSearchCV for Weight Decay value***

![7](https://user-images.githubusercontent.com/41862477/49699366-32f14780-fbf6-11e8-8a57-88509a33a7c8.JPG)

> ***GridSearchCV for Dropout value***

![8](https://user-images.githubusercontent.com/41862477/49699367-32f14780-fbf6-11e8-9d87-af9ebe9aad7f.JPG)

***

***Thanks for going through this post! Any feedbacks are duly appreciated.***
