# Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates

###Abstract: 
This post describe a phenomenon, which goes by name “super-convergence”, where neural networks can be trained an order of magnitude faster than with standard training methods. One of the key elements of super-convergence is training with one cycle policy and a large maximum learning rate. A primary insight that allows super-convergence training is that large learning rates regularize the training, hence requiring a reduction of all other forms of regularization in order to preserve an optimal regularization balance. 

###Motivation: 
There is a reasonable point of view that says that training a model to 94% test accuracy on CIFAR10 is a meaningless exercise since state-of-the-art is above 98%. "State-of-the-art" accuracy is an ill-conditioned target in the sense that throwing a larger model, more hyperparameter tuning, more data augmentation or longer training time at the problem will typically lead to accuracy gains, making fair comparison between different works a delicate task. Moreover, existence of super-convergence is relevant to understanding why deep networks generalize well. Fig1 illustrates the super-convergence: 



