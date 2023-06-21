# Evaluation

We evaluate three identical training runs of each model and take the average metrics. This is to ensure reliable estimate of performance 

Structure your directories as below:
```
Evaluation
│   README.md
└───*Combined Dataset* ('JSRT_Padchest', 'Montgomery_Shenzen')
    └───Predictions
    │   └───*Model Name* ('HybridGNet', 'UNet', 'Joint', etc)
    │        └───1
    │        └───2
    │        └───3
    └───Scores
