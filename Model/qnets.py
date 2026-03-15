import torch
import torch.nn as nn
import brevitas.nn as qnn

class QuantTinyCNN(nn.Module):
    def __init__(self, num_classes, bit_width):
        super(QuantTinyCNN, self).__init__()
        
        # Microcontroller suitable architecture: small number of channels, aggressive pooling
        self.layer1 = nn.Sequential(
            qnn.QuantConv2d(1, 16, kernel_size=3, stride=1, padding=1, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            qnn.QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            qnn.QuantConv2d(32, 32, kernel_size=3, stride=1, padding=1, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            nn.MaxPool2d(kernel_size=2)
        )
        
        # 28x28 -> 14x14 -> 7x7 -> 3x3
        self.fc = nn.Sequential(
            qnn.QuantLinear(32 * 3 * 3, 64, weight_bit_width=bit_width),
            qnn.QuantReLU(bit_width=bit_width),
            qnn.QuantLinear(64, num_classes, weight_bit_width=bit_width)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def qtinycnn(num_classes, bit_width):
    return QuantTinyCNN(num_classes, bit_width)
