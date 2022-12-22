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
            # self.linear = nn.utils.spectral_norm(nn.Linear(inc, outc, bias=usebias))
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
        if self.dropout != None:
            out = self.dropout(out)
        return out

class Generator(nn.Module):
    def __init__(self, gauss_dim, feature_dim):
        super().__init__()

        self.mlp = nn.Sequential(
            LinearLayer(gauss_dim, feature_dim),
            LinearLayer(feature_dim, feature_dim),
            # LinearLayer(feature_dim, feature_dim),
            LinearLayer(feature_dim, feature_dim),
            LinearLayer(feature_dim, feature_dim, activation=False),
            nn.LayerNorm([feature_dim]),
        )

        self.mlp2 = nn.Sequential(
            nn.Conv1d(1, 2, 17, padding=8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(2, 4, 17, padding=8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(4, 8, 17, padding=8),
            nn.LeakyReLU(0.2),
            nn.Conv1d(8, 1, 17, padding=8),
            # nn.LeakyReLU(0.2),
        )
        self.ll = LinearLayer(gauss_dim, feature_dim, activation=False)

    def forward(self, x):
        # x = F.normalize(x)
        # feature = self.mlp(x)
        out = self.ll(x)
        out2 = out.view([out.shape[0], 1, out.shape[1]])
        feature = self.mlp2(out2)
        feature = feature.view([out.shape[0], out.shape[1]])
        
        return feature

if __name__ == "__main__":
    net = Generator(128, 512)
    net.eval()
    
    gauss_noise = torch.randn((2, 128))
    for i in range(10):
        out = net(gauss_noise)
        print(out.shape)