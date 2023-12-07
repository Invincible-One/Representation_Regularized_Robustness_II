import os

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F



class ElasticNetLoss(nn.Module):
    def __init__(self, base_loss, alpha, rho, log_path=None):
        super(ElasticNetLoss, self).__init__()
        if base_loss == "CE":
            self.base_loss = F.cross_entropy
        else:
            raise NotImplementedError

        self.alpha = alpha
        self.rho = rho
        self.log_path = log_path

        self.reset_monitor()

    def forward(self, outputs, targets, representations):
        loss = self.base_loss(outputs, targets)

        repr_dim = representations.size(0)
        representations = representations.view(repr_dim, -1)
        _, s, _ = torch.svd(representations)
        #WARNING: didn't normalize
        avg_nuclear_norm = torch.sum(s)
        avg_frobenius_norm = torch.linalg.norm(representations, 'fro')

        #WARNING: debugging purposes
        #print(f"Debugging! avg nuclear norm: {avg_nuclear_norm.item()}, avg frobenius norm: {avg_frobenius_norm.item()}")

        total_loss = loss + self.alpha * self.rho * avg_nuclear_norm + self.alpha * (1 - self.rho)  * avg_frobenius_norm

        self.update_monitor(nuclear_norm=avg_nuclear_norm, frobenius_norm=avg_frobenius_norm)
        #COMMENT: l1 and l2 regularizations
        #l1_norms = torch.norm(representations, p=1, dim=tuple(range(1, representations.dim())))
        #l2_norms = torch.norm(representations, p=2, dim=tuple(range(1, representations.dim())))
        #l1_norm = l1_norms.mean()
        #l2_norm = l2_norms.mean()
        #total_loss = loss + self.l1_strength * l1_norm + self.l2_strength * l2_norm

        return total_loss

    def reset_monitor(self):
        self.cumulative_nuclear_norm = 0
        self.cumulative_frobenius_norm = 0
        self.batch_count = 0
    
    def update_monitor(self, nuclear_norm, frobenius_norm):
        self.cumulative_nuclear_norm += nuclear_norm.cpu().item()
        self.cumulative_frobenius_norm += frobenius_norm.cpu().item()
        self.batch_count += 1

    def display(self, epoch_counter, split):
        avg_nuclear = self.cumulative_nuclear_norm / self.batch_count if self.batch_count else 0
        avg_frobenius = self.cumulative_frobenius_norm / self.batch_count if self.batch_count else 0
        data = {
                "epoch_info": [f"Epoch{epoch_counter}_{split}"],
                "avg_nuclear": [avg_nuclear],
                "avg_frobenius": [avg_frobenius],
                }
        df = pd.DataFrame(data)
        write_header = not os.path.exists(self.log_path)

        if self.log_path is not None:
            df.to_csv(self.log_path, mode='a', index=False, header=write_header)

        self.reset_monitor()


def get_loss(args):
    return ElasticNetLoss(args.base_loss, args.alpha, args.rho, args.log_path)
