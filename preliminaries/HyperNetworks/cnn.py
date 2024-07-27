# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class TemplateBank(nn.Module):
    def __init__(self, num_templates, in_planes, out_planes, kernel_size):
        super(TemplateBank, self).__init__()
        self.coefficient_shape = (num_templates, 1, 1, 1, 1)
        templates = [
            torch.Tensor(out_planes, in_planes, kernel_size, kernel_size)
            for _ in range(num_templates)
        ]
        # print(f"{templates[0].size()}")
        # print(num_templates)
        for i in range(num_templates):
            init.kaiming_normal_(templates[i])
        self.templates = nn.Parameter(torch.stack(templates))

    def forward(self, coefficients):
        # print(coefficients.size()) # torch.Size([4, 1, 1, 1, 1])
        # print(f"{(self.templates * coefficients).shape}")
        return (self.templates * coefficients).sum(0)


class SConv2d(nn.Module):
    def __init__(self, bank, stride=1, padding=1):
        super(SConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.bank = bank
        self.coefficients = nn.Parameter(torch.zeros(bank.coefficient_shape))

    def forward(self, input):
        params = self.bank(self.coefficients)
        # print(f"{params.size()}") # torch.Size([320, 320, 3, 3])
        return F.conv2d(input, params, stride=self.stride, padding=self.padding)


class Block(nn.Module):
    def __init__(self, in_planes, out_planes, stride, bank=None):
        super(Block, self).__init__()
        self.bank = bank

        self.bn1 = nn.BatchNorm2d(in_planes)
        if self.bank:
            self.conv1 = SConv2d(self.bank)
        else:
            self.conv1 = nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )

        self.bn2 = nn.BatchNorm2d(out_planes)
        if self.bank:
            self.conv2 = SConv2d(self.bank)
        else:
            self.conv2 = nn.Conv2d(
                out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False
            )

        self.relu = nn.ReLU(inplace=True)
        self.equalInOut = in_planes == out_planes
        self.convShortcut = (
            (not self.equalInOut)
            and nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False,
            )
            or None
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        if not self.equalInOut:
            residual = out
        out = self.conv2(self.relu(self.bn2(self.conv1(out))))
        if self.convShortcut is not None:
            residual = self.convShortcut(residual)
        return out + residual


class SWRN(nn.Module):
    def __init__(self, depth, width, num_templates, num_classes):
        super(SWRN, self).__init__()

        n_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        layers_per_bank = 2 * (num_blocks - 1)
        print(
            "SWRN : Depth : {} , Widen Factor : {}, Templates per Group : {}".format(
                depth, width, num_templates
            )
        )

        self.num_classes = num_classes
        self.num_templates = num_templates

        self.conv_3x3 = nn.Conv2d(
            3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False
        )

        self.bank_1 = TemplateBank(self.num_templates, n_channels[1], n_channels[1], 3)
        self.stage_1 = self._make_layer(
            n_channels[0], n_channels[1], num_blocks, self.bank_1, 1
        )

        self.bank_2 = TemplateBank(self.num_templates, n_channels[2], n_channels[2], 3)
        self.stage_2 = self._make_layer(
            n_channels[1], n_channels[2], num_blocks, self.bank_2, 2
        )

        self.bank_3 = TemplateBank(self.num_templates, n_channels[3], n_channels[3], 3)
        self.stage_3 = self._make_layer(
            n_channels[2], n_channels[3], num_blocks, self.bank_3, 2
        )

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True)
        )
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(n_channels[3], num_classes)

        for i in range(1, 4):
            coefficient_inits = torch.zeros(
                (layers_per_bank, num_templates, 1, 1, 1, 1)
            )
            nn.init.orthogonal_(coefficient_inits)
            # sconv_group = filter(lambda (name, module): isinstance(module, SConv2d) and "stage_%s"%i in name, self.named_modules())
            sconv_group = filter(
                lambda name_module: isinstance(name_module[1], SConv2d)
                and f"stage_{i}" in name_module[0],
                self.named_modules(),
            )
            for j, (name, module) in enumerate(sconv_group):
                module.coefficients.data = coefficient_inits[j]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, in_planes, out_planes, num_blocks, bank, stride=1):
        blocks = []
        blocks.append(Block(in_planes, out_planes, stride))
        for i in range(1, num_blocks):
            blocks.append(Block(out_planes, out_planes, 1, bank))
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def swrn(depth, width, num_templates, num_classes=10):
    model = SWRN(depth, width, num_templates, num_classes)
    return model


# %%
if __name__ == "__main__":
    input = torch.randn(1, 3, 32, 32)
    model = swrn(28, 10, 4)
    output = model(input)
    print(output.size())
# %%
