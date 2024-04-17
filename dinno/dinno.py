import os
from torch import optim, nn, utils, Tensor
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import lightning as L
from util import create_graph
from copy import deepcopy
import torch
from agent import Agent
import torch.nn.functional as F
import numpy as np

# From the code
class MNISTConvNet(nn.Module):
    """Implements a basic convolutional neural network with one
    convolutional layer and two subsequent linear layers for the MNIST
    classification problem.
    """

    def __init__(self, num_filters=3, kernel_size=5, linear_width=64):
        super().__init__()
        conv_out_width = 28 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        self.seq = nn.Sequential(
            nn.Conv2d(1, num_filters, kernel_size, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc1_indim, linear_width),
            nn.ReLU(inplace=True),
            nn.Linear(linear_width, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        return self.seq(x)

# define the LightningModule
class DiNNO(L.LightningModule):
    def __init__(self,
                 agent_config,
                 graph_type="complete",
                 fiedler_value=None,
                 num_classes=10,
                 B=2,
                 rho=0.5,
                 lr_start=0.005,
                 lr_finish=0.0005,
                 oits=2000,
                 rho_update=0.0003):
        super().__init__()
        self.num_nodes = len(agent_config)
        self.G, self.G_connectivity = create_graph(num_nodes = self.num_nodes,
                                                   graph_type = graph_type,
                                                   target_connectivity = fiedler_value)
        
        base_model = MNISTConvNet()
        self.agent_id_to_idx = {agent["id"]: i for i, agent in enumerate(agent_config)}

        # Initialize the networks for each agent
        self.agent_config = agent_config
        self.agents = nn.ModuleDict({agent["id"]: Agent(config=agent_config[i],
                                          model=deepcopy(base_model),
                                          idx=i)
                                          for i, agent in enumerate(self.agent_config)})
        
        # Set automatic optimization to false
        self.automatic_optimization = False
        self.criterion = torch.nn.NLLLoss()
        self.rho = rho
        
        self.lr_schedule = torch.linspace(
                lr_start,
                lr_finish,
                oits,
            )
        
        self.manual_global_step=0
        
        self.save_hyperparameters()

    def calculate_loss(self, x, y, theta_reg, curr_agent):
        primal_loss = self.criterion(curr_agent.model(x), y)
        theta = torch.nn.utils.parameters_to_vector(curr_agent.model.parameters())
        reg = torch.sum(torch.square(torch.cdist(theta.reshape(1, -1), theta_reg)))
        return primal_loss + torch.dot(curr_agent.dual, theta) + self.rho * reg
        
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward

        
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            curr_agent.set_flattened_params()
        self.rho *= (1 + self.hparams.rho_update) 

        # Technically this can be done in parallel
        all_losses = []
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            neighbor_indices = self.G.neighbors(curr_agent.idx)
            neighbor_params = torch.stack([self.agents[self.agent_config[idx]['id']].get_flattened_params() for idx in neighbor_indices])
            theta = curr_agent.get_flattened_params()
            
            # DUAL VARIABLE UPDATE
            curr_agent.dual += self.rho * (theta - neighbor_params).sum(0)
            theta_reg = (theta + neighbor_params) / 2
            curr_batch = batch[curr_agent.idx]
            x, y = curr_batch
            split_size = x.shape[0] // self.hparams.B
            
            # Optimize for current agent
            opt = optim.Adam(curr_agent.parameters(), lr=self.lr_schedule[self.manual_global_step])
            
            self.log(f"learning_rate", self.lr_schedule[self.manual_global_step], logger=True)
            
            # Loop through B times for the current agent
            for tau in range(self.hparams.B):
                
                # PRIMAL UPDATE
                x_split = x[tau*split_size:(tau+1)*split_size]
                y_split = y[tau*split_size:(tau+1)*split_size]
                opt.zero_grad()
                loss = self.calculate_loss(x_split, y_split, theta_reg, curr_agent=curr_agent)
                self.manual_backward(loss)
                opt.step()

            self.log(f"{agent_id}_train_loss", loss, logger=True)
            all_losses.append(loss.item())
        self.log(f"train_loss", np.mean(all_losses), logger=True, prog_bar=True)
        
        self.manual_global_step += 1

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        all_accs = []
        for agent_id in self.agents:
            curr_agent = self.agents[agent_id]
            logits = curr_agent.model(x)
            preds = torch.argmax(logits, dim=1)
            acc = torch.sum(y == preds).float() / len(y)
            self.log(f'{agent_id}_val_acc', acc.item(), on_epoch=True, logger=True)
            all_accs.append(acc.item())
        self.log(f'val_acc', np.mean(all_accs), on_epoch=True, prog_bar=True, logger=True)


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

        
        