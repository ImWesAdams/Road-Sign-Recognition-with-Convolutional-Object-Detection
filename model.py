import torch

class down_block(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size = 3, stride = 2, dropout = 0.1, bias = True):
        super().__init__()
        """Main downconvolution block with residual"""
        # Primary block structure
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.Conv2d(input_channels, output_channels, kernel_size = kernel_size, padding = kernel_size//2, stride = stride, bias = bias),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size = kernel_size, padding = kernel_size//2, bias = bias),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(output_channels, output_channels, kernel_size = kernel_size, padding = kernel_size//2, bias = bias)
        )

        # Residual connection
        self.downsample = None
        if input_channels != output_channels or stride != 1:
            self.downsample = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, output_channels, kernel_size = 1, padding = 0, stride = stride),
            torch.nn.BatchNorm2d(output_channels)
        ) 
        
        # Determining dropout from input parameters      
        self.dropout = torch.nn.Dropout2d(dropout)
            
    def forward(self, x):
        """Forward pass through the downconvolution"""
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        fin = self.net(x) + identity
        return self.dropout(fin)

class up_block(torch.nn.Module):
    def __init__(self, input_channels, output_channels, dropout = 0.1):
        super().__init__()
        """Main upconvolution block with residual"""
        # Upconvolution
        self.upc = torch.nn.Sequential(
        torch.nn.ConvTranspose2d(input_channels, output_channels, kernel_size = 2, stride = 2)
        )

        # Processing block
        self.res = torch.nn.Sequential(
            down_block(input_channels, output_channels, stride = 1)
        )

        # Determining dropout from input parameters
        self.dropout = torch.nn.Dropout2d(dropout)

    def forward(self, x, skip):
        """Forward pass through the upconvolution"""
        zu = self.upc(x)
        # print(zu.shape, skip.shape)
        zuc = torch.cat([zu, skip], axis = 1)
        fin = self.res(zuc)
        return self.dropout(fin)


class Detector(torch.nn.Module):        
    def __init__(self, input_channels = 3, output_channels = 1, stride = 2, dropout = 0.1, residuals = True):
        super().__init__()
        """Make a UNet with residuals detector"""

        # First input downblock
        self.d1 = torch.nn.Sequential(
            torch.nn.BatchNorm2d(input_channels),
            torch.nn.Conv2d(input_channels, 16, kernel_size = 7, padding = 7//2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size = 3, padding = 3//2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 16, kernel_size = 3, padding = 3//2)
        )
        # The rest of the downblocks
        self.d2 = down_block(16, 32, stride = stride)
        self.d3 = down_block(32, 64, stride = stride)
        self.d4 = down_block(64, 128, stride = stride)
        self.d5 = down_block(128, 256, stride = stride)

        # The upblocks
        self.u1 = up_block(256, 128)
        self.u2 = up_block(128, 64)
        self.u3 = up_block(64, 32)
        self.u4 = up_block(32, 16)

        # Final convolution for prediction
        self.fc = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 3, stride = 1, padding = 3//2),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels = 8, out_channels = output_channels, kernel_size = 3, stride = 1, padding = 3//2),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.Dropout2d(dropout),
            torch.nn.Conv2d(in_channels = output_channels, out_channels = output_channels, kernel_size = 1),
        )


    def forward(self, x):
        """Make a forward pass through the detector."""
        # Downblock passes
        a1 = self.d1(x)
        # print('a1', a1.shape)
        a2 = self.d2(a1)
        # print('a2', a2.shape)
        a3 = self.d3(a2)
        # print('a3', a3.shape)
        a4 = self.d4(a3)
        # print('a4', a4.shape)
        a5 = self.d5(a4)
        # print('a5', a5.shape)

        # Upblock passes
        b1 = self.u1(a5, a4)
        # print('b1', b1.shape)
        b2 = self.u2(b1, a3)
        # print('b2', b2.shape)
        b3 = self.u3(b2, a2)
        # print('b3', b3.shape)
        b4 = self.u4(b3, a1)
        # print('b4', b4.shape)

        # Final prediction
        out = self.fc(b4)
        # print('out', out.shape)
        return out

def save_model(model, step):
    from torch import save
    from os import path
    final = str(step) + '_det.th'
    # print(path.join(path.dirname(path.abspath(__file__))))
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), final))


def load_model(place, output_channels = 1):
    from torch import load
    from os import path
    r = Detector(output_channels = output_channels)
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), place), map_location='cpu'))
    return r