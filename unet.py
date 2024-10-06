import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self,input_channels,output_channels):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            # first convolution
            nn.Conv2d(in_channels=input_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels), # we set bias false because Batch normalization will cancel it out
            nn.ReLU(inplace=True),
            # second convolution
            nn.Conv2d(in_channels=output_channels,
                      out_channels=output_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(output_channels), # we set bias false because Batch normalization will cancel it out
            nn.ReLU(inplace=True),
        )
    def forward(self,x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self,input_channels=3,output_channels=1,features = [64,128,256,512]):
        super(UNet,self).__init__()
        self.upsample=nn.ModuleList()
        self.downsample = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)

        # downsampling step
        for feature in features:
            self.downsample.append(DoubleConv(input_channels=input_channels,output_channels=feature))
            input_channels = feature 
        
        # upsampling step
        for feature in reversed(features):
            self.upsample.append(nn.ConvTranspose2d(in_channels=feature*2,out_channels=feature,kernel_size=2,stride=2))
            self.upsample.append(DoubleConv(input_channels=feature*2,output_channels=feature))
        
        # bottleneck layer
        self.bottleneck = DoubleConv(features[-1],features[-1]*2)
        self.output_conv = nn.Conv2d(features[0],out_channels=output_channels,kernel_size=1)
    
    def forward(self,x):
        # we start with downsampling step
        skip_connections = []
        for down in self.downsample:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # then the bottleneck layer comes
        x = self.bottleneck(x)
        # the first element in the skip connections belongs to highest order so we reverse it
        skip_connections = skip_connections[::-1] # we want to go backwards in that order when doing concatenation

         
        # now we perform upsampling
        for idx in range(0,len(self.upsample),2):  # up-doubleconv - one step, so we take two steps in a loop
            x = self.upsample[idx](x)
            skip_connection = skip_connections[idx//2]
            if x.shape != skip_connection.shape:
                x = TF.resize(x,size=skip_connection.shape[2:]) # take height and width only
            concat_skip = torch.cat((skip_connection,x),dim=1)
            x = self.upsample[idx+1](concat_skip)
        return self.output_conv(x)

def test_network():
    x = torch.randn((3,1,160,160))
    model = UNet(input_channels=1,output_channels=1)
    preds = model(x)
    print(preds.shape)
    print(x.shape)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test_network()
