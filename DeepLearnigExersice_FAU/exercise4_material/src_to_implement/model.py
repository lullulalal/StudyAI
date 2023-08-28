import torch as t
import torchvision.models as models

class ResNet(t.nn.Module):
    def __init__(self):
        super().__init__()
        #resnet = models.resnext50_32x4d(pretrained=True)
        resnet = models.resnet50(pretrained=True)
        #resnet = models.wide_resnet50_2(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.layer3.parameters():
            param.requires_grad = True
        #for param in resnet.layer2.parameters():
            #param.requires_grad = True
        #for param in resnet.layer1.parameters():
        #    param.requires_grad = True

        self.pretrained = t.nn.Sequential(*(list(resnet.children())[:-1]))
        self.dropout = t.nn.Dropout2d(0.4, inplace=True)
        self.fc = t.nn.Linear(2048, 2)
        self.sigmoid = t.nn.Sigmoid()

    def forward(self, x):
        #x = self.avg_pool(x)
        x = self.pretrained(x)
        x = t.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class DenseNet(t.nn.Module):
    def __init__(self):
        super().__init__()
        densenet = models.densenet169(pretrained=True)

        num_ftrs = densenet.classifier.in_features
        densenet.classifier = t.nn.Sequential(
            t.nn.Dropout2d(0.4, inplace=True),
            t.nn.Linear(num_ftrs, 2),
            t.nn.Sigmoid()
        )
        self.pretrained = densenet

    def forward(self, x):
        x = self.pretrained(x)
        return x

class CustomModel(t.nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()
        self.densenet = DenseNet()

    def forward(self, x):
        pred1 = self.resnet(x)
        pred2 = self.densenet(x)

        y = (pred1 + pred2) / 2
        return y

'''
class ResNet(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.inplanes = 64  # input feature map

        self.conv1 = torch.nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.inplanes)
        self.relu = torch.nn.ReLU(inplace=True)
        self.max_pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.resblock1 = self.BasicBlock(64, 64, 1)
        self.resblock2 = self.BasicBlock(64, 128, 2)
        self.resblock3 = self.BasicBlock(128, 256, 2)
        self.resblock4 = self.BasicBlock(256, 512, 2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(512, 2)
        self.sigmoid = torch.nn.Sigmoid()

        self.dropout = torch.nn.Dropout2d(p=0.2, inplace=True)

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 1)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        zero_init_residual = True
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, self.BasicBlock):
                    torch.nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        #out = self.dropout(out)
        out = self.relu(out)
        out = self.max_pool(out)
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = self.resblock4(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.dropout(out)
        out = self.fc(out)
        out = self.sigmoid(out)

        return out

    class BasicBlock(torch.nn.Module):

        def __init__(self, in_channels, out_channels, stride):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
            self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

            self.bn1 = torch.nn.BatchNorm2d(out_channels)
            self.bn2 = torch.nn.BatchNorm2d(out_channels)

            self.relu1 = torch.nn.ReLU(inplace=True)
            self.relu2 = torch.nn.ReLU(inplace=True)

            self.dropout1 = torch.nn.Dropout2d(p=0.2, inplace=True)
            self.dropout2 = torch.nn.Dropout2d(p=0.2, inplace=True)

            self.shortcut = torch.nn.Sequential()
            if stride != 1:
                self.shortcut = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    torch.nn.BatchNorm2d(out_channels)
                )

            for m in self.modules():
                if isinstance(m, torch.nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
                    torch.nn.init.constant_(m.weight, 1)
                    torch.nn.init.constant_(m.bias, 0)

        def forward(self, x):
            out = self.conv1(x)
            out = self.bn1(out)
            #out = self.dropout1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += self.shortcut(x)
            #out = self.dropout2(out)
            out = self.relu2(out)

            return out
'''
