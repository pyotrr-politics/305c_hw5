import torch
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

torch.manual_seed(305)
# os.chdir('C:/Users/letra/iCloudDrive/수업자료/2024/Stats 300/HW')


### Import preprocessed data
from preprocess import responses_test, responses_train
from preprocess import covariates_test, covariates_train

shares_test = torch.tensor(responses_test["share"], dtype=torch.float32).to(device)
shares_train = torch.tensor(responses_train["share"], dtype=torch.float32).to(device)

covariates_test = torch.tensor(covariates_test.to_numpy(dtype=float), dtype=torch.float32).to(device)
covariates_train = torch.tensor(covariates_train.to_numpy(dtype=float), dtype=torch.float32).to(device)



### Import Neural Network Architecture
from architecture import share_fitter
from architecture import device

torch.manual_seed(300)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Global hyperparameters
SMALL_ITERS = 4000
LARGE_ITERS = 10000
EVAL_ITERS = 100
covar_size = covariates_test.shape[1]



### Implement CV
from sklearn.model_selection import KFold

learning_rate = 1e-4
kf = KFold(n_splits=10, shuffle=True, random_state=300)

pred_upper = torch.empty(len(shares_test), 1).to(device)
pred_lower = torch.empty(len(shares_test), 1).to(device)
loss_list = []

for rest_index, fold_index in kf.split(shares_train):
    pred, losses = share_fitter(fold_index, rest_index, covar_size, learning_rate)
    loss_list.append([losses['train'].detach().item(), losses['val'].detach().item()])

    resid = torch.abs(pred(covariates_train[fold_index, ])[0].squeeze() - shares_train[fold_index])
    pred_test = pred(covariates_test)[0]
    
    pred_upper = torch.hstack([pred_upper, pred_test + resid])
    pred_lower = torch.hstack([pred_lower, pred_test - resid])
    

pred_upper = torch.quantile(pred_upper[:, 1:], 0.9, dim=1).squeeze()
pred_lower = torch.quantile(pred_lower[:, 1:], 0.1, dim=1).squeeze()

loss = torch.mean(torch.vstack([torch.tensor(loss) for loss in loss_list]), dim=0)
print("train RMSE: ", round(math.sqrt(loss[0].item()), 4), 
      " test RMSE: ", round(math.sqrt(loss[1].item()), 4))







### Implement conformal prediction
width = pred_upper - pred_lower
is_covered = (pred_lower <= shares_test) & (shares_test <= pred_upper)

print("average width: ", round(torch.mean(width.to(dtype=torch.float64)).detach().item(), 4))
print("coverage rate: ", round(torch.mean(is_covered.to(dtype=torch.float64)).detach().item(), 4))

# plot 
shares_test = pd.Series(shares_test.detach().cpu().numpy())
sorted_shares = shares_test.sort_values()
pred_upper = pred_upper[shares_test.sort_values().index].detach().cpu().numpy()
pred_lower = pred_lower[shares_test.sort_values().index].detach().cpu().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(x=list(range(len(sorted_shares))), y=sorted_shares,
         color='#ff4444', s=1)
plt.errorbar(x=list(range(len(sorted_shares))), 
             y=(pred_lower+pred_upper)/2,             
             yerr=(pred_upper-pred_lower)/2, 
             fmt='none',                      
             capsize=1,                    # length of error bar caps
             capthick=.5,                   # thickness of error bar caps
             elinewidth=.5,                 # thickness of error bar lines
             markersize=1,                 # size of dots
             color='grey',                 # color of dots and error bars
             alpha=0.2)                    # transparency

plt.title('Prediction Intervals', fontsize=14)
plt.xlabel('County', fontsize=12)
plt.ylabel('Democratic vote share', fontsize=12)
plt.grid(True, alpha=0.3)
plt.xlim([0, len(sorted_shares)])

plt.savefig('300C pset5_q3.pdf', 
            format='pdf',
            dpi=300,           # Higher DPI for better quality
            bbox_inches='tight',  # Tight bounding box
            pad_inches=0.1)    # Padding around the plot

plt.close()