import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class T5LayerNorm_(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style. No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1,hidden_size,1))
        self.variance_epsilon = eps

    def forward(self, hidden_states):

        # T5 uses a layer_norm which only scales and doesn't shift, which is also known as Root Mean
        # Square Layer Normalization https://arxiv.org/abs/1910.07467 thus varience is calculated
        # w/o mean and there is no bias. Additionally we want to make sure that the accumulation for
        # half-precision inputs is done in fp32

        variance = hidden_states.to(torch.float32).pow(2).mean(1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into half-precision if necessary
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        return self.weight * hidden_states

class BasicConv1d_LN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(BasicConv1d_LN, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, bias=True)
        self.ln = T5LayerNorm_(out_channels)
        self.kernel_size = kernel_size

    def adjust_mask(self, x, masks):
        # Trim or pad masks to match the third dimension of x
        if masks.shape[2] < x.shape[2]:
            print('extended mask')
            padding = masks[:, :, :x.shape[2]].new_zeros(
                masks.shape[:2] + (masks.shape[2] - x.shape[2],),
                dtype=torch.bool,
            )
            masks = torch.cat((masks, padding), dim=2)
        elif masks.shape[2] > x.shape[2]:
            print('trimmed mask')
            masks = masks[:, :, :x.shape[2]]
        return masks

    def forward(self,x,masks):
        out = x
        if self.kernel_size > 1:
           masks = self.adjust_mask(x, masks)
           out = torch.where(masks,out,torch.zeros(size=(1,),device=out.device))
        out = self.conv(out)
        out = F.hardswish(out)
        if out.shape == x.shape:
           out = self.ln(out) + x
        else:
           out = self.ln(out)
        return out  

class Convolution_Predictor(nn.Module):
    def __init__(self, num_channels):
        super(Convolution_Predictor, self).__init__()
        self.l1conv1d = BasicConv1d_LN(num_channels, num_channels, 1)
        self.l2conv1d = BasicConv1d_LN(num_channels//2, num_channels//2, 3)
        self.fc = nn.Linear(num_channels, 3)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(1)

        self.output = BasicConv1d_LN(num_channels, 8, 1)

    def channel_split(self, t):
        m = t.shape[1] // 2
        t1 = t[:, :m, :]
        t2 = t[:, m:, :]

        return(t1, t2)
    
    def channel_merge(self, t1, t2):
        return torch.cat((t1, t2), dim=1)

    def adjust_mask(self, x, masks):
        # Trim or pad masks to match the third dimension of x
        if masks.shape[2] < x.shape[2]:
            print('extended mask')
            padding = masks[:, :, :x.shape[2]].new_zeros(
                masks.shape[:2] + (masks.shape[2] - x.shape[2],),
                dtype=torch.bool,
            )
            masks = torch.cat((masks, padding), dim=2)
        elif masks.shape[2] > x.shape[2]:
            print('trimmed mask')
            masks = masks[:, :, :x.shape[2]]
        return masks
    
    def forward(self, x, masks):
        masks = self.adjust_mask(x, masks)
        out = self.l1conv1d(x, masks)

        out1, out2 = self.channel_split(out)
        out1 = self.dropout(out1)
        out1 = self.l2conv1d(out1, masks)

        out = self.channel_merge(out1, out2)
        out = self.dropout(out)

        out = self.output(out, masks)
        final_mask = masks[:, :8, :]
        out = torch.where(final_mask,out,torch.zeros(size=(1,),device=out.device))
        
        return out
    
    
