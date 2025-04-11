# NX-414-Team-16
Project for the 2025 NX-414 - Brain-like computation and intelligence course


Week7-------------------------------------------

Shallow CNN:            var=0.1773 / corr=0.4125

Week8-------------------------------------------

-----------------Optimize-----------------------

sched=0.6 + norm        var=0.2015 / corr=0.4442

same + dropout=0.2      var=0.0649 / corr=0.2548

same + dropout=0.2 + L2 var=0.2015 / corr=0.4510

------------Finetune existing model-------------

Finetune ResNet:        var=0.4467 / corr=0.6560

Finetune EffNet:        var=0.4677 / corr=0.6725

Finetune vit_base:      var=0.0585 / corr=0.2185

Finetune vit_tiny:      var=0.1353 / corr=0.3682

------------Two objectives-----------------------

Biobj ResNet:           var=0.2360 / corr=0.4760

Biobj ResNet (0.3 lab): var=0.2984 / corr=0.5346


Here is what I did and the decisions I took: 

First, I tried to improve the shallowCNN model. I tried to modify the normalization, the dropout, the scheduler and add a weight decay. Here is the logical chain-of-thoughts I used.

----------------------------------------------

---> For all the models that I trained, I loaded pretrained weigths and add a small linear transformation layer in order to adapt to the activity dimension

Second of all for the finetuning part, I wanted to finetune a few models. I wanted to test CNN models and Vision transformer. I started with pretrained ResNet50 and it gave good results. So I decided to move to a slightly bigger and more recent model: pretrained EffectiveNet. It gave even better results. Then I tested Vision models. I first tried to use vit_base and the results were pretty bad. I explained it by the model to be too big for the data we have. So I moved on with a smaller version of a vision transformer model. I tried the "small" version and the "tiny" version. The "tiny" version gave best results as "small" and "base" version but was not as good CNN models. I excluded the "small" version to not pollute the code.

Intermediate conclusion: pretrained weights of existing models seem to give really good results for CNN architecture. It does not give good results for vision transformers (worst than shallow CNN).

----------------------------------------------

---> Here I also added a linear transfomation to the models. But I had to add two different heads because there are two objectives with label dimension and activity dimension I had to adapt to. 

Third, I look into training the models with two obectives. I chose ResNet50 as the model to train on and I used pretrained weights, motivated by the results of the first part (it seems that finetuning pretrained weights is the most promising so far). One objective is the same objective with the MSELoss. The other objective is the prediction of the labels of the images with the CrossEntropyLoss. In my first training the importance of both objectives is the same and it leads to worst results than simply finetuning. My hypothesis was then that the more we also try to learn the labels, the worst the model becomes to predict neural activity. To test my hypothesis, I trained a model for which the weight that takes the neural activity objective is more than three time bigger than the label objective. This valided my hypothesis: the results were in-between predicting only the neural activity and the balanced version of training with two objectives. 

Update: I also try to train on bionjective without using the pretrained weigts. This leads to worst performance: var=0.0968 / corr=0.3020. I didn't include this neither to not pollute the table/the code.

Intermediate conclusion: Using two objectives, with one trying to predict the labels, doesn't help the model. It has the oposit effect, namely: the more we try to train on the labels, the worst the model becomes.

