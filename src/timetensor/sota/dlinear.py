## Adapted from https://github.com/honeywell21/DLinear/blob/main/models/DLinear.py

import torch
import torch.nn as nn

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)
        x = self.avg(x)
        return x

class series_decomp(nn.Module):
    def __init__(self, kernel_size=25):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, lags, dim, horizon, kernel_size=25):
        super(DLinear, self).__init__()
        self.lags, self.dim, self.horizon  = lags, dim, horizon

        self.decomposition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.ModuleList()
        self.Linear_Trend = nn.ModuleList()
        
        for i in range(self.dim):
            self.Linear_Seasonal.append(nn.Linear(self.lags,self.horizon))
            self.Linear_Trend.append(nn.Linear(self.lags,self.horizon))

    def forward(self, x, context=None): #x (bs, dim, lags)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.horizon],dtype=seasonal_init.dtype).to(seasonal_init.device)
        trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.horizon],dtype=trend_init.dtype).to(trend_init.device)
        for i in range(self.dim):
            seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
            trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        
        x = seasonal_output + trend_output
        return x