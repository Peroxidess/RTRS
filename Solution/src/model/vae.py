import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


class VAutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(VAutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.dim_input_list = [dim_input, max(dim_input // 2, 4), max(dim_input // 3, 4), z_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_input_list[0], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.dim_input_list[1]),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
            nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(self.dim_input_list[2], z_dim)
        self.fc_logvar = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.dim_input_list[2]),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.BatchNorm1d(self.dim_input_list[2]),
            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),
        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, c=None):
        z = self._encode(x)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        # z = F.normalize(z, p=2, dim=1)
        x_recon = self._decode(z)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec, mu, logvar, beta=1e-8):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD


class AutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(AutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.dim_input_list = [dim_input, max(dim_input // 2, 4), max(dim_input // 3, 4), z_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_input_list[0], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(self.dim_input_list[1]),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
            nn.ReLU(True),

        )
        self.fc_z = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, self.dim_input_list[2]),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.BatchNorm1d(self.dim_input_list[2]),
            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),
        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, c=None):
        z = self._encode(x)
        z_ = self.fc_z(z)
        z_ = F.normalize(z_, p=2, dim=1)
        x_recon = self._decode(z_)
        return x_recon, z_, None, None

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec, mu=None, logvar=None, beta=None):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        return MSE


class CAutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(CAutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.emb_output = 4
        self.dim_input_list = [dim_input, max(dim_input // 2, 4), max(dim_input // 3, 4), z_dim]
        self.embedding = Embedding_self(dim_input=dim_input * 2, emb_input=1, emb_output=self.emb_output)
        self.embedding_reverse = Embedding_Reverse(self.emb_output, 1)
        self.encoder = nn.Sequential(

            nn.Linear(self.dim_input_list[0] + 1, self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
            nn.ReLU(True),
        )
        self.fc_z = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(

            nn.Linear(z_dim + 1, self.dim_input_list[2]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),
        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, c):
        c = c.view(-1, 1)
        c_ = c.repeat(x.shape[0], 1)
        x_cat = torch.cat([x, c_], dim=1)
        x_emb = self.embedding(x_cat)
        z = self._encode(x_emb)
        z_ = self.fc_z(z)
        z_ = F.normalize(z_, p=2, dim=1)
        z_flatten = torch.flatten(z_, 1)
        z_cat = torch.cat([z_, torch.unsqueeze(x_emb[:, :, -1], -1)], dim=-1)
        x_de = self._decode(z_cat)
        x_recon = self.embedding_reverse(x_de)
        return x_recon, z_flatten, None, None

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec, mu=None, logvar=None, beta=None):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        return MSE


class Embedding_self(nn.Module):
    def __init__(self, dim_input, emb_input=1, emb_output=4):
        super(Embedding_self, self).__init__()
        self.fnn_list = []
        for _ in range(dim_input):
            self.fnn_list.append(nn.Linear(emb_input, emb_output))

    def forward(self, x):
        for i in range(x.shape[1]):
            x_emb = self.fnn_list[i](x[:, [i]])
            x_emb = x_emb.unsqueeze(1)
            if i == 0:
                x_emb_all = x_emb
            else:
                x_emb_all = torch.cat((x_emb_all, x_emb), 1)
        return x_emb_all.permute(0, 2, 1)


class Embedding_Reverse(nn.Module):
    def __init__(self, emb_input=4, emb_output=1):
        super(Embedding_Reverse, self).__init__()
        self.fnn_data_l1 = nn.Linear(emb_input, emb_input)
        self.fnn_data_l2 = nn.Linear(emb_input, emb_output)

    def forward(self, x):
        x_ = x.permute(0, 2, 1)
        x_out1 = F.gelu(self.fnn_data_l1(x_))
        x_out2 = F.gelu(self.fnn_data_l2(x_out1))
        x_out = torch.squeeze(x_out2)
        return x_out


class CVAutoEncoder(nn.Module):
    def __init__(self, dim_input, z_dim=32, seed=2022):
        super(CVAutoEncoder, self).__init__()
        self.seed = seed
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.z_dim = z_dim
        self.dim_input = dim_input
        self.dim_input_list = [dim_input, max(dim_input // 2, 4), max(dim_input // 4, 4), z_dim]
        self.encoder = nn.Sequential(
            nn.Linear(self.dim_input_list[0] + 1, self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[2]),
        )
        self.fc_mu = nn.Linear(self.dim_input_list[2], z_dim)
        self.fc_logvar = nn.Linear(self.dim_input_list[2], z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim + 1, self.dim_input_list[2]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[2], self.dim_input_list[1]),
            nn.ReLU(True),
            nn.Linear(self.dim_input_list[1], self.dim_input_list[0]),
        )
        self.weight_init()
        self.to(device)

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    kaiming_init(m)
            except:
                kaiming_init(block)

    def forward(self, x, c):
        c = c.view(-1, 1)
        c_ = c.repeat(x.shape[0], 1)
        x_cat = torch.cat([x, c_], dim=1)
        z = self._encode(x_cat)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        z_cat = torch.cat([z, c_], dim=1)
        x_recon = self._decode(z_cat)
        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        if mu.is_cuda:
            stds, epsilon = stds.cuda(), epsilon.cuda()
        latents = epsilon * stds + mu
        return latents

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

    def loss(self, x, x_rec, mu, logvar, beta=1e-8):
        mse_loss = nn.MSELoss()
        MSE = mse_loss(x, x_rec)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD
