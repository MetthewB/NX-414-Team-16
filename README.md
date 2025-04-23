# NX-414-Team-16
Project for the 2025 NX-414 - Brain-like computation and intelligence course


Week7-------------------------------------------

Shallow CNN:            var=0.1773 / corr=0.4125

Week8-------------------------------------------

-----------------Optimize-----------------------

sched=0.6 + norm        var=0.2015 / corr=0.4442

same + dropout=0.2      var=0.0649 / corr=0.2548

same + dropout=0.2 + L2 var=0.2015 / corr=0.4510

same + add layer->model var=0.2044 / corr=0.4574

------------Finetune existing model-------------

Finetune ResNet:        var=0.4467 / corr=0.6560

Finetune EffNet:        var=0.4677 / corr=0.6725

Finetune vit_base:      var=0.0585 / corr=0.2185

Finetune vit_tiny:      var=0.1353 / corr=0.3682

------------Two objectives-----------------------

Biobj ResNet:           var=0.2360 / corr=0.4760

Biobj ResNet (0.3 lab): var=0.2984 / corr=0.5346


Here is what we did and the decisions we took: 

First, we tried to improve the shallowCNN model. We tried to modify the normalization, the dropout, the scheduler and add a weight decay. Here is the logical chain-of-thoughts we used. First; we thought that using the mean and variance of the data instead of the one from ResNet could be beneficial. Hence we modified this. But with the new normalization, the model was not learning fast enough: we had to change the scheduler. It worked well. Next, we tried to modify the dropout to 0.2 thinking that it would help the model to be more expressive. It didn't work well. But this combined with L2 normalization was beneficial. Finally, we concluded that it would be hard to improve hyperparameters more, so we updated the model to add one convolutional layer. It lead to some little improvement. we concluded that making the model more expressive was beneficial, hence the next section. 

----------------------------------------------

---> For all the models that we trained, we loaded pretrained weigths and add a small linear transformation layer in order to adapt to the activity dimension

Second of all for the finetuning part, we wanted to finetune a few models. We wanted to test CNN models and Vision transformer. We started with pretrained ResNet50 and it gave good results. So we decided to move to a slightly bigger and more recent model: pretrained EffectiveNet. It gave even better results. Then we tested Vision models. We first tried to use vit_base and the results were pretty bad. We explained it by the model to be too big for the data we have. So we moved on with a smaller version of a vision transformer model. We tried the "small" version and the "tiny" version. The "tiny" version gave best results as "small" and "base" version but was not as good CNN models. We excluded the "small" version to not pollute the code.

Intermediate conclusion: pretrained weights of existing models seem to give really good results for CNN architecture. It does not give good results for vision transformers (worst than shallow CNN).

----------------------------------------------

---> Here we also added a linear transfomation to the models. But we had to add two different heads because there are two objectives with label dimension and activity dimension we had to adapt to. 

Third, we look into training the models with two obectives. We chose ResNet50 as the model to train on and we used pretrained weights, motivated by the results of the first part (it seems that finetuning pretrained weights is the most promising so far). One objective is the same objective with the MSELoss. The other objective is the prediction of the labels of the images with the CrossEntropyLoss. In our first training the importance of both objectives is the same and it leads to worst results than simply finetuning. Our hypothesis was then that the more we also try to learn the labels, the worst the model becomes to predict neural activity. To test our hypothesis, we trained a model for which the weight that takes the neural activity objective is more than three time bigger than the label objective. This valided our hypothesis: the results were in-between predicting only the neural activity and the balanced version of training with two objectives. 

We also try to train on bi-objective without using the pretrained weigts. This leads to worst performance: var=0.0968 / corr=0.3020. We didn't include this to not pollute the table and the code.