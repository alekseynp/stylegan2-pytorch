import torch
import torch.nn as nn


class RandomDo(nn.Module):
    def __init__(self, prob_do, thing):
        super(RandomDo, self).__init__()
        self.prob_do = prob_do
        self.thing = thing

    def forward(self, x):
        batch_size = x.size()[0]

        mask = torch.rand(batch_size) < self.prob_do

        x[mask] = self.thing(x[mask])
        return x
