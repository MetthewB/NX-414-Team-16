# NX-414-Team-16
Project for the 2025 NX-414 - Brain-like computation and intelligence course
(Arnault Stähli, Matthieu Beylard-Ozeroff, Francesco Ceccucci)

Please download our DenseNet201 here: https://drive.google.com/drive/folders/1RMFtDjMh7rq9Nh-BxOjJjslNARcA2nBk?usp=sharing

This .zip file contains: 

- test.ipynb: this file is shows the results of our best model
- code.ipynb: the code for all the models we tried to train to predict the neural activity
- Report.pdf: the detail of our investigation methods and our findings
- models/DenseNet201.pth: the weights of our best model. Those are used to produce the results of test.ipynb that evaluates our DenseNet201 model. The model is not in the folder because of the size limit of this folder. It is accessible here: https://drive.google.com/drive/folders/1RMFtDjMh7rq9Nh-BxOjJjslNARcA2nBk?usp=sharing
- utils.py: the file containing some helper functions for the notebook code.ipynb

Important notes: 

We set seeds in our notebook code.ipynb in order to make our code reproductible. Two important notes: 

1. The results are environment dependant. Even with the seeds, running the code on a different envrionment might lead to different results. Specifically, we used the T4 GPU on Google Colab to run everything up to part "Utilize different pretrained models with the task-driven modeling approach". For the part "Utilize different pretrained models with the task-driven modeling approach", we used the Nvidia A100 on Google Colab.

2. For the first part of week 8 ("Optimize your current models by adjusting hyperparameters and implementing different regularizations"), the training sometimes crashed due to non-convergence in gradient descent. The seed allowed us to make sure it does not happen when reruning the notebook. However, merging week 6, week 7 and week 8 might cause the problem to happen again. To solve the issue, loading the librairies and running the code for week 8 directly should solve any related issue. 
