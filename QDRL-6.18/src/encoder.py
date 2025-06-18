import copy
from typing import Optional
import math
import torch.nn.functional as F
from torch import nn, Tensor
import torch

class Transformer_encoder(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=4,
                dim_feedforward=1024, dropout=0.1,
                activation="relu", normalize_before=False,
                use_query=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)#1024是拼接后的维度
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.d_model = d_model
        self.use_query = use_query

        self.cls_token = nn.Parameter(torch.randn(1, d_model))  # 可训练的CLS嵌入

        self.encoder = TransformerEncoder(d_model, encoder_layer, num_encoder_layers, encoder_norm)
        # 位置编码
        self.pos_emb = PositionalEncoding(d_model)

        self._reset_parameters()


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, v, q, src_key_padding_mask, mask):#, query_embed, pos_embed
        # 将CLS嵌入复制到每个样本的批次中
        cls_token = self.cls_token.expand(v.size(0), -1, -1)  # 扩展CLS到批次大小 #20%
        if self.use_query:
            pass
        else:
            q = cls_token

        v = torch.cat([v, q], dim=1)  # 将CLS嵌入添加到序列前

        # 位置编码
        pos = self.pos_emb(v)

        # 2. 更新 src_key_padding_mask：加一个 False 表示 CLS 是有效的
        cls_mask = torch.zeros((src_key_padding_mask.size(0), 1), dtype=torch.bool, device=src_key_padding_mask.device)
        src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)  # [B, T+1]

        # print('encoder',src_key_padding_mask.shape)
        v = self.encoder(v, src_key_padding_mask=src_key_padding_mask, mask=mask, pos=pos)
        return v


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers,)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                ):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)#

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)

        if src_mask is not None and src_mask.dim() == 3:
            batch_size, seq_len, _ = src_mask.size()
            src_mask = src_mask.unsqueeze(1).repeat(1, self.self_attn.num_heads, 1, 1)
            src_mask = src_mask.reshape(batch_size * self.self_attn.num_heads, seq_len, seq_len)

        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)

        if src_mask is not None and src_mask.dim() == 3:
            batch_size, seq_len, _ = src_mask.size()
            src_mask = src_mask.unsqueeze(1).repeat(1, self.self_attn.num_heads, 1, 1)
            src_mask = src_mask.reshape(batch_size * self.self_attn.num_heads, seq_len, seq_len)

        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


############位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, d_embed, seq_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_embed
        pe = torch.zeros(seq_len, d_embed)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_embed, 2).float()
            * (-math.log(10000.0) / d_embed)
        )
        pe[:, 0::2] = torch.sin(position * div_term)# 字嵌入维度为偶数时
        pe[:, 1::2] = torch.cos(position * div_term)# 字嵌入维度为奇数时
        pe = pe.unsqueeze(0)## 在指定维度0上增加维度大小为1[3,4] -> [1,3,4]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x = x * math.sqrt(self.d_model)# sqrt() 方法返回数字x的平方根 适应多头注意力机制的计算，它说“这个数值的选取略微增加了内积值的标准差，从而降低了内积过大的风险，可以使词嵌入张量在使用多头注意力时更加稳定可靠”
        x_pos = self.pe[:, : x.size(1), :]# 变为x.size(1)长度，torch.Size([1, 4, 512])
        return x_pos#layer层会再加上位置信息


if __name__ == '__main__':
    batch_size = 2
    seq_len = 4
    input_dim = 16
    model_dim = 32
    num_heads = 4
    num_layers = 2

    model = Transformer_encoder(d_model=input_dim, nhead=2, num_encoder_layers=4,dim_feedforward=256)

    # 输入数据
    x = torch.randn(batch_size, seq_len, input_dim)

    q = torch.randn(batch_size, 1, input_dim)

    attn_mask = torch.zeros(batch_size, seq_len+1, seq_len+1)
    attn_mask[:, :, 1] = float('-inf')  # 举例

    print(attn_mask.shape)
    output = model(x, q, attn_mask)
