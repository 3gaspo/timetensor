## Adapted from: https://github.com/yuqinie98/PatchTST

from torch import nn

from .PatchTST_backbone import PatchTST_backbone


class PatchTST(nn.Module):
    def __init__(self, context_window = 24*7, target_window=24, **kwargs):
        super().__init__()
        
        max_seq_len = context_window
        d_k, d_v = None, None
        attn_dropout = 0.
        act = "gelu"
        key_padding_mask = "auto"
        padding_var, attn_mask = None, None
        res_attention = True
        pre_norm, store_attn = False, False
        pe = "zeros"
        learn_pe = True
        pretrain_head = False
        head_type = "flatten"
        verbose = False

        c_in = 1
        n_layers = 3
        n_heads = 16
        d_model = 128
        d_ff = 256
        dropout = 0.2
        fc_dropout = 0.2
        head_dropout = 0.0
        individual = 0
        patch_len = 16
        stride = 8
        padding_patch = "end"

        
        self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, attn_dropout=attn_dropout,
                                dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                pretrain_head=pretrain_head, head_type=head_type, individual=individual, verbose=verbose, **kwargs)
    
    
    def forward(self, x, c=None):  # x: [Batch, Channel, Input length]
        x = self.model(x)
        return x