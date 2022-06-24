
import torch
from torch import nn

class LinearST(nn.Module):

    def __init__(self, sim_queue_length, quantile_count, in_features=1, out_features=2):
        super(LinearST, self).__init__()

        self.sim_queue_length = sim_queue_length
        self.quantile_count = quantile_count

        self.quantiles = torch.zeros(out_features, quantile_count)

        self.is_first_forward_pass = True

        self.layer = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):

        x = self.layer(x)
        if self.is_first_forward_pass:
            self.is_first_forward_pass = False
            for i in range(x.size(dim=1)):
                self.quantiles[i, :] = torch.quantile(x[:, i], torch.tensor([i / (self.quantile_count + 1) for i in range(1, self.quantile_count + 1)]))
        else:
            x_cube = torch.unsqueeze(x, dim=2).expand(x.size(dim=0), x.size(dim=1), self.quantile_count)
            quantile_cube = torch.unsqueeze(self.quantiles, dim=0).expand(x_cube.size())

            x_is_over_mask = (x_cube >= quantile_cube).int()
            x_is_under_mask = torch.ones_like(x_is_over_mask) - x_is_over_mask

            # TODO cache portions of the k_cube so that it is not recalculated on each forward pass
            k_cube = torch.tensor([[[i for i in range(self.quantile_count)]]]).expand(x_cube.size())

            # s_cube should be fibers of (self.quantiles[:, -1] - self.quantiles[:, 1]) / 10 / (self.quantile_count + 2)
            # repeated in the second and third dimension
            s_cube = torch.unsqueeze(torch.unsqueeze(self.quantiles[:, -1] - self.quantiles[:, 1], dim=1), dim=0).expand(k_cube.size())
            s_cube = s_cube / 10 / (self.quantile_count + 2)

            self.quantile_changes = x_is_over_mask * (torch.ones_like(k_cube, dtype=torch.float) + k_cube)
            self.quantile_changes -= x_is_under_mask * (self.quantile_count * torch.ones_like(k_cube) - k_cube)
            self.quantile_changes *= s_cube / (self.quantile_count + 1)

            self.quantile_changes = torch.mean(self.quantile_changes, dim=0)

            self.quantiles += self.quantile_changes

        return x
            

