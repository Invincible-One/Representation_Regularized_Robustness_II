import torch
import torch.nn as nn
import torch.nn.functional as F



class ElasticNetLoss(nn.Module):
    def __init__(self, base_loss, nuclear_strength, frobenius_strength):
        super(ElasticNetLoss, self).__init__()
        if base_loss == "CE":
            self.base_loss = F.cross_entropy
        else:
            raise NotImplementedError

        self.nuclear_strength = nuclear_strength
        self.frobenius_strength = frobenius_strength

    def forward(self, outputs, targets, representations):
        loss = self.base_loss(outputs, targets)

        repr_dim = representations.size(0)
        representations = representations.view(repr_dim, -1)
        _, s, _ = torch.svd(representations)
        avg_nuclear_norm = torch.sum(s) / repr_dim
        avg_frobenius_norm = torch.linalg.norm(representations, 'fro') / repr_dim

        #WARNING: debugging purposes
        print(f"Debugging! avg nuclear norm: {avg_nuclear_norm.item()}, avg frobenius norm: {avg_frobenius_norm.item()}")

        total_loss = loss + self.nuclear_strength * avg_nuclear_norm + self.frobenius_strength * avg_frobenius_norm

        #COMMENT: l1 and l2 regularizations
        #l1_norms = torch.norm(representations, p=1, dim=tuple(range(1, representations.dim())))
        #l2_norms = torch.norm(representations, p=2, dim=tuple(range(1, representations.dim())))
        #l1_norm = l1_norms.mean()
        #l2_norm = l2_norms.mean()
        #total_loss = loss + self.l1_strength * l1_norm + self.l2_strength * l2_norm

        return total_loss



def get_loss(args):
    return ElasticNetLoss(args.base_loss, args.nuclear_strength, args.frobenius_strength)
