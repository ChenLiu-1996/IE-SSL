import torch
from sklearn.metrics import accuracy_score


class NTXentLoss(torch.nn.Module):

    def __init__(self, temperature: float = 1.0):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.epsilon = 1e-7

    def forward(self, z1: torch.Tensor, z2: torch.Tensor):
        assert z1.shape == z2.shape
        B, _ = z1.shape

        loss = 0
        z1 = torch.nn.functional.normalize(input=z1, p=2, dim=1)
        z2 = torch.nn.functional.normalize(input=z2, p=2, dim=1)

        # Create a matrix that represent the [i,j] entries of positive pairs.
        # Diagonal (self) are positive pairs.
        pos_pair_ij = torch.diag(torch.ones(B))
        pos_pair_ij = pos_pair_ij.bool()

        # Similarity matrix.
        sim_matrix = torch.matmul(z1, z2.T)

        # Entries noted by 1's in `pos_pair_ij` are similarities of positive pairs.
        numerator = torch.sum(
            torch.exp(sim_matrix[pos_pair_ij] / self.temperature))

        # Entries elsewhere are similarities of negative pairs.
        denominator = torch.sum(
            torch.exp(sim_matrix[~pos_pair_ij] / self.temperature))

        loss += -torch.log(numerator /
                           (denominator + self.epsilon) + self.epsilon)

        pseudo_acc = accuracy_score(
            y_true=pos_pair_ij.cpu().detach().numpy().reshape(-1),
            y_pred=sim_matrix.cpu().detach().numpy().reshape(-1) > 0.5)

        return loss / B, pseudo_acc
