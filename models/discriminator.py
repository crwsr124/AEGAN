
from paddle2torch import *

class LinearLayer(nn.Module):
    def __init__(self, inc, outc, activation=True, weight_scale=False, weight_norm=True, dropout=False):
        super().__init__()
        self.w_lr = 1.0 / math.sqrt(inc) if weight_scale else None

        usebias = True
        self.bias = None
        if weight_scale:
            usebias = False
            # self.bias = nn.Parameter( torch.randn(outc) )

        if weight_norm:
            # self.linear = nn.utils.spectral_norm(nn.Linear(inc, outc))
            self.linear = nn.utils.weight_norm(nn.Linear(inc, outc), dim=None)
        else:
            self.linear = nn.Linear(inc, outc)
        # nn.init.kaiming_normal_(self.linear.weight, nonlinearity="leaky_relu")
        nn.init.orthogonal_(self.linear.weight)

        self.activation = None
        if activation:
            self.activation = nn.LeakyReLU(0.2)

        self.dropout = None
        if dropout:
            self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        out = self.linear(x)
        if self.bias != None:
            out = out * self.w_lr
            bias = self.bias.repeat(x.shape[0], 1)
            out = out + bias
        if self.activation != None:
            out = self.activation(out)
        # if self.dropout != None:
        #     out = self.dropout(out)
        return out

class BatchDiscriminator(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            LinearLayer(feature_dim, feature_dim, dropout=False),
            LinearLayer(feature_dim, feature_dim, dropout=False),
            LinearLayer(feature_dim, feature_dim, dropout=False, activation=False),
        )
        self.single_logit = nn.Sequential(
            LinearLayer(feature_dim, feature_dim, dropout=True),
            LinearLayer(feature_dim, feature_dim, dropout=True),
            LinearLayer(feature_dim, 1, dropout=False, activation=False),
        )
        self.union_batch = 2
        self.union_logit = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(self.union_batch, self.union_batch*2, 1), dim=None),
            # nn.utils.spectral_norm(nn.Conv1d(self.union_batch, self.union_batch*2, 1)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.utils.weight_norm(nn.Conv1d(self.union_batch*2, 1, 1), dim=None),
            # nn.utils.spectral_norm(nn.Conv1d(self.union_batch*2, 1, 1)),
            nn.LeakyReLU(0.2),
            # nn.Dropout(p=0.2),
            nn.Flatten(),

            LinearLayer(feature_dim, feature_dim, dropout=True),
            LinearLayer(feature_dim, feature_dim, dropout=True),
            LinearLayer(feature_dim, 1, dropout=False, activation=False),
        )

        self.norm = nn.LayerNorm([feature_dim])

    def forward(self, x):
        # x = self.encoder(x)
        # x = self.norm(x)
        # feature = layer_norm(feature)
        # x_noise = x + torch.randn((x.shape[0], x.shape[1]))
        x_noise = x + 0.1*torch.randn((x.shape[0], x.shape[1]))
        single_logit = self.single_logit(x_noise)

        x_noise = x_noise.view((x_noise.shape[0]//self.union_batch, self.union_batch, -1))
        union_logit = self.union_logit(x_noise)
        # print("111111", union_logit)
        # union_logit = union_logit.repeat(self.union_batch, 1)
        union_logit = union_logit.repeat_interleave(self.union_batch, 0)
        # union_logit = union_logit.tile([self.union_batch, 1]) #should not use tile
        # print("222222", union_logit)

        # logit = torch.cat([union_logit, single_logit], dim=1)
        logit = torch.concat([union_logit, single_logit], 1)
        
        return logit

if __name__ == "__main__":
    net = BatchDiscriminator(512)
    net.eval()
    
    feature = torch.randn((2, 512))
    for i in range(1000):
        out = net(feature)
        print(out.shape)