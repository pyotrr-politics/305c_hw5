import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim

import pandas as pd
from jaxtyping import Float, Int
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'


### Neural Network Architecture
class FeedForward(nn.Module):
    """ 
    neural network with three hidden layers
    R^p |-> [0, 1]
    """

    def __init__(self, covar_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(covar_size, 6 * covar_size),
            nn.ReLU(),
            nn.Linear(6 * covar_size, 4 * covar_size),
            nn.ReLU(),
            nn.Linear(4 * covar_size, 2 * covar_size),
            nn.ReLU(),
            nn.Linear(2 * covar_size, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))
    

class SharePredictor(nn.Module):
    def __init__(self, covar_size):
        """
        Args:
            covar_size: int, the number of covariates to predict vote shares with (p)
        """
        super().__init__()
        self.covar_transform = FeedForward(covar_size)

        # good initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, covar_df, targets=None):
        """
        Args:
            covar_df: tensor of integers, shape (N, p)
            targets: tensor of integers, provides shares we are predicitng, shape (N)
        """
        pred_shares = self.covar_transform(covar_df)
        if targets is None:
            loss = None
        else:
            loss = torch.mean((pred_shares - targets) ** 2)
        return pred_shares, loss




### model fitters
@torch.no_grad()
def estimate_loss(model, eval_iters, fold_index, rest_index):
    """
    Args:
      model: model being evaluated
      eval_iters: number of batches to average over
      context_window_size: size of the context window
      device: 'cpu' or 'cuda' (should be 'cuda' if available)
    """
    out = {}
    for split in ['train', 'val']:
        covars = covariates_train[rest_index, ] if split == 'train' else covariates_train[fold_index, ]
        responses = shares_train[rest_index] if split == 'train' else shares_train[fold_index]

        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            _, loss = model(covars, responses)
            losses[k] = loss.item()
        out[split] = losses.mean()
    return out


def share_fitter(fold_index, rest_index, covar_size, learning_rate):
    pred = SharePredictor(covar_size).to(device)
    optimizer = torch.optim.AdamW(pred.parameters(), lr=learning_rate)
    eval_interval = LARGE_ITERS/10
    loss_list = []

    for it in tqdm(range(LARGE_ITERS)):
        # every once in a while evaluate the loss on train and val sets
        if it % eval_interval == 0 or it == LARGE_ITERS - 1:
            print(f"iteration {it}")
            losses = estimate_loss(pred, EVAL_ITERS, fold_index, rest_index)
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        _, loss = pred(covariates_train[rest_index, ], shares_train[rest_index])
        loss_list.append(loss.detach().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    losses = estimate_loss(pred, EVAL_ITERS, fold_index, rest_index)
    return pred, losses