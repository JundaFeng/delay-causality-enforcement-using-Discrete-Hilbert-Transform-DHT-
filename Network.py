import torch

class Net(torch.nn.Module):
    def __init__(self, in_dim, n_hidden_1, out_dim):
        super(Net, self).__init__()
        self.layer1 = torch.nn.Linear(in_dim, n_hidden_1) #torch.nn.Sequential(torch.nn.Linear(in_dim, n_hidden_1), torch.nn.Tanh())
        self.act = torch.nn.Tanh()
        self.layer2 = torch.nn.Linear(n_hidden_1, out_dim) #torch.nn.Sequential(torch.nn.Linear(n_hidden_1, out_dim))#, torch.nn.Tanh())
        # self.layer3 = torch.nn.Sequential(torch.nn.Linear(n_hidden_2, out_dim), torch.nn.Tanh())
        # self.layer4 = torch.nn.Sequential(torch.nn.Linear(n_hidden_3, n_hidden_4), torch.nn.Tanh())
        # self.layer4 = torch.nn.Sequential(torch.nn.Linear(n_hidden_3, ))

    def forward(self, x):
        x = self.layer1(x)
        x = self.act(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        # x = self.layer5(x)
        return x