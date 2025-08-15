
import torch
import torch.nn as nn
import torch.nn.functional as F


class FC_layer(nn.Module):
    def __init__(self, args):
        super(FC_layer, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.out = nn.Linear(128, args.num_classes)
        self.drop = nn.Dropout(p=args.dropout_rate)
        self.droput = args.dropout_rate

    def forward(self, x):
        
        if self.droput > 0:
            x = self.drop(F.relu(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.out(x)
        return x
    def get_weights(self):
        return self.out.weight
    

class GradReverse(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg() * ctx.constant
        return grad_output, None

    def grad_reverse(x, constant):
        return GradReverse.apply(x, constant)

class Grad(torch.autograd.Function):
    """
    Extension of grad reverse layer
    """
    @staticmethod
    def forward(ctx, x, constant):
        ctx.constant = constant
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output * ctx.constant
        return grad_output, None

    def grad(x, constant):
        return Grad.apply(x, constant)


class Domain_classifier_DG(nn.Module):

    def __init__(self, num_class, encode_dim, dropout_rate=0.5):
        super(Domain_classifier_DG, self).__init__()

        self.num_class = num_class
        self.encode_dim = encode_dim
        self.dropout = dropout_rate
        self.drop = nn.Dropout(p=dropout_rate)

        self.L1 = nn.Linear(encode_dim, 500, bias=False)
        self.bn1 = nn.BatchNorm1d(500)

        self.L2 = nn.Linear(500, 500, bias=False)
        self.bn2 = nn.BatchNorm1d(500)
        self.out = nn.Linear(500, num_class)
        

    def forward(self, input, constant, Reverse):

        if Reverse:
            input = GradReverse.grad_reverse(input, constant)
        else:
            input = Grad.grad(input, constant)
        
        if self.dropout > 0:
            x = self.drop(F.relu(self.bn1(self.L1(input))))
            x = self.drop(F.relu(self.bn2(self.L2(x))))
        else:
            x = F.relu(self.bn1(self.L1(input)))
            x = F.relu(self.bn2(self.L2(x)))

        logits = self.out(x)

        return logits 
    



class ConvBlock(nn.Module):
    """Convolutional module: Conv1D + BatchNorm + (optional) ReLU."""

    def __init__(
        self,
        n_in_channels: int,
        n_out_channels: int,
        kernel_size: int,
        padding_mode: str = "replicate",
        include_relu: bool = True,
    ) -> None:
        super().__init__()
        layers = [
            nn.Conv1d(
                in_channels=n_in_channels,
                out_channels=n_out_channels,
                kernel_size=kernel_size,
                padding="same",
                padding_mode=padding_mode,
            ),
            nn.BatchNorm1d(num_features=n_out_channels),
        ]
        if include_relu:
            layers.append(nn.ReLU())
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_block(x)
        return out


def manual_pad(x: torch.Tensor, min_length: int) -> torch.Tensor:
    """
    Manual padding function that pads x to a minimum length with replicate padding.
    PyTorch padding complains if x is too short relative to the desired pad size, hence this function.

    :param x: Input tensor to be padded.
    :param min_length: Length to which the tensor will be padded.
    :return: Padded tensor of length min_length.
    """
    # Calculate amount of padding required
    pad_amount = min_length - x.shape[-1]
    # Split either side
    pad_left = pad_amount // 2
    pad_right = pad_amount - pad_left
    # Pad left (replicate first value)
    pad_x = F.pad(x, [pad_left, 0], mode="constant", value=x[:, :, 0].item())
    # Pad right (replicate last value)
    pad_x = F.pad(pad_x, [0, pad_right], mode="constant", value=x[:, :, -1].item())
    return pad_x


class ResNetFeatureExtractor(nn.Module):
    """ResNet feature extractor implementation for use in MILLET. Same as original architecture."""

    def __init__(self, n_in_channels: int, padding_mode: str = "replicate"):
        super().__init__()
        self.n_in_channels = n_in_channels
        self.instance_encoder = nn.Sequential(
            ResNetBlock(n_in_channels, 64, padding_mode=padding_mode),
            ResNetBlock(64, 128, padding_mode=padding_mode),
            ResNetBlock(128, 128, padding_mode=padding_mode),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # PyTorch doesn't like replicate padding if the input tensor is too small, so pad manually to min length
        min_len = 5
        x = x.permute(0, 2, 1)
        if x.shape[-1] >= min_len:
            x=  self.instance_encoder(x)
        else:
            padded_x = manual_pad(x, min_len)
            x =  self.instance_encoder(padded_x)
        
        return self.pool(x).squeeze(-1)


class ResNetBlock(nn.Module):
    """ResNet block of three convolutional blocks with different kernel sizes."""

    def __init__(self, in_channels: int, out_channels: int, padding_mode: str = "replicate") -> None:
        super().__init__()

        # Create layers
        layers = []
        for block_idx, kernel_size in enumerate([8, 5, 3]):
            in_c = in_channels if block_idx == 0 else out_channels
            include_relu = block_idx == 2
            conv_block = ConvBlock(in_c, out_channels, kernel_size, padding_mode, include_relu)
            layers.append(conv_block)
        self.layers = nn.Sequential(*layers)

        # Create residual
        self.residual: nn.Module
        if in_channels != out_channels:
            residual_layers = [
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    padding="same",
                    padding_mode=padding_mode,
                ),
                nn.BatchNorm1d(num_features=out_channels),
            ]
            self.residual = nn.Sequential(*residual_layers)
        else:
            self.residual = nn.BatchNorm1d(num_features=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.layers(x) + self.residual(x))


class FeatureExtracter(nn.Module):
    def __init__(self, N_channels):
        super(FeatureExtracter, self).__init__()
        self.conv1 = nn.Conv1d(N_channels, 128, kernel_size=8, stride=2, bias=False)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, bias=False)
        self.bn2 = nn.BatchNorm1d(256)
        
        self.conv3 = nn.Conv1d(256, 128, kernel_size=3, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward( self, x):
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = self.pool(x)

        x = x.reshape(x.shape[0], x.shape[1])
        return x   
    
# x = torch.randn(16, 90,77).cuda()
# model = FeatureExtracter(77).cuda()
# out = model(x)
# print(out.shape)

