I compared 4 kinds of loss function(L1 Loss,CE Loss,Focal Loss(gamma = 0.5),Focal Loss(gamma = 2)).
We observed that when we used MAE (L1) loss as the loss function, the model consistently performed consistently and poorly on the test set, with an accuracy of only 0.105, indicating that MAE (L1) loss performed poorly on this issue.

CE Loss,Focal Loss(gamma = 0.5),Focal Loss(gamma = 2)both have better accuracy on this isssue, end up with accuracy of around 0.57.

I test 3 learning rate(1e-1, 1e-2, 1e-3),1e-2 is a better choice for this issue.