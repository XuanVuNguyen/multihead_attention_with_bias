from typing import Optional, Tuple
import torch
from torch import nn, Tensor
import torch.nn.functional as F

class MultiHeadAttentionWithBias(nn.Module):
    '''
    Multihead Attention as in "Attention is All You Need", but modified by adding a bias term before softmax when calculating attention weights.
    '''
    def __init__(
        self,
        embed_dim: int, # embed_dim is the same as input query dim
        num_heads: int,
        proj_bias: bool=True,
        attention_dropout: float=0.0,
        kv_dim: Optional[int]=None,):
        '''
        Init args:
          embed_dim (int): the embed_dim of input query and the output state of this module.
          num_heads (int): numbers of attention heads.
        '''
        
        super().__init__()        
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim*num_heads==embed_dim, "num_heads must be divisible by embed_dim"
        self.num_heads = num_heads
        
        self.scale = self.head_dim**(-0.5)
        
        self.embed_dim = embed_dim
        self.kv_dim = embed_dim if kv_dim is None else kv_dim
        
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=proj_bias)
        self.W_k = nn.Linear(self.kv_dim, embed_dim, bias=proj_bias)
        self.W_v = nn.Linear(self.kv_dim, embed_dim, bias=proj_bias)
        self.combine_heads = nn.Linear(embed_dim, embed_dim, bias=proj_bias)
        
        self.attention_dropout = attention_dropout
        
        self.init_parameters()
        
    def init_parameters(self):
        qkv_same_dim = self.embed_dim==self.kv_dim
        gain = 2**(-0.5) if qkv_same_dim else 1
        [nn.init.xavier_uniform_(layer.weight, gain=gain) for layer in [self.W_q, self.W_k, self.W_v]]
        nn.init.xavier_uniform_(self.combine_heads.weight)
        for layer in self.children():
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0.)
        
    def forward(
        self, 
        query: Tensor, 
        key_value: Optional[Tensor]=None, 
        # value: Optional[Tensor]=None,
        past_keys_values: Optional[Tuple[Tensor]]=None,
        attention_bias: Optional[Tensor]=None,
        attention_mask: Optional[Tensor]=None,
        is_autoregressive: bool=False,
        return_attn_weights: bool=False,
        return_current_keys_values: bool=False,):
        '''
        Self-attention is performed when:
         *Both :key_value: and :past_keys_values: are set to None.
         *:key_value: is set to None and :is_autoregressive: is True.
        Args:
         :query, key_value: `(bsz, len, fea_dim)`: :key_value: will be ignored if :is_regression: is `False` and
             :past_keys_values: is specified.
             
         :past_keys_values: store the projected keys and values to prevent re-projecting when calling this layer
             multiple times but not changing keys and values (think about the cross_attention in decoder when running inference).
             Can be updated with the newly projected keys and values if :is_autoregressive: is True (self_attention in decoder when running inference).
             
         :attention_bias: `(bsz, q_len, kv_len)`
         :attention_mask: `(bsz, q_len, kv_len)`: masking out the tokens of key that we dont want to attend to. Masked tokens are labeled as 0.
         :is_autoregressive: whether to update the :past_keys_values: with the current projected keys and values.
         :return_attn_weights: (bool): if True then return the attention weights of all heads along with the query state.
         :return_current_keys_values: (bool): if True then return the current projected keys and values pair.
        '''
        query = query.transpose(0, 1) # switch to time_dim_first
        key_value = key_value.transpose(0, 1) if key_value is not None else query
        
        q_len, bsz, embed_dim = query.size()
        
        assert self.kv_dim == key_value.size(2) and self.embed_dim == embed_dim
        
        q = self.W_q(query)
        
        if past_keys_values is not None:
            assert len(past_keys_values)==2
            past_k = past_keys_values[0].transpose(0, 1)
            past_v = past_keys_values[1].transpose(0, 1) # time dim first
        else:
            past_k, past_v = (torch.zeros((0, bsz, embed_dim), dtype=q.dtype),
                         torch.zeros((0, bsz, embed_dim), dtype=q.dtype))
                
        if is_autoregressive:
            k = self.W_k(key_value)
            v = self.W_v(key_value)
            k = torch.cat([past_k, k], dim=0)
            v = torch.cat([past_v, v], dim=0)
        else:
            if past_keys_values is None:
                k = self.W_k(key_value)
                v = self.W_v(key_value)
            else:
                k, v = past_k, past_v
               
        # up til now k and v are time dim first
        kv_len = k.size(0)
        assert kv_len == v.size(0), "key and value need to have the same len"
        assert bsz == k.size(1) and bsz == v.size(1), "query, key and value need to have the same batch size"
        
        current_keys_values = [k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous()] if return_current_keys_values else None
        
        q = q.contiguous().view(q_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(kv_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(kv_len, bsz*self.num_heads, self.head_dim).transpose(0, 1)
        # Lesson learned: only reshape the two dims that are adjacent to each other.
        
        attn_weight = torch.bmm(q, k.transpose(1, 2)) # `(bsz*self.num_heads, q_len, kv_len)`
        # scale
        attn_weight *= self.scale

        # modify multihead attention by adding a bias term
        if attention_bias is not None:
            attn_weight = attn_weight.view(bsz, self.num_heads, q_len, kv_len)
            attn_weight += attention_bias.unsqueeze(1)
            attn_weight = attn_weight.view(bsz*self.num_heads, q_len, kv_len)
            
        # apply key mask
        if attention_mask is not None:
            attn_weight = attn_weight.view(bsz, self.num_heads, q_len, kv_len)
            attn_weight = attn_weight.masked_fill(
                (1-attention_mask.unsqueeze(1)).to(torch.bool), # unsqueeze to broadcast
                float("-inf"),
            )
            attn_weight = attn_weight.view(bsz*self.num_heads, q_len, kv_len)
        
        # softmax
        attn_weight = F.softmax(attn_weight, dim=-1) # `(bsz*self.num_heads, q_len, kv_len)`
        attn_weight = nn.functional.dropout(attn_weight, self.attention_dropout, training=self.training)
        
        # multiplying attention weight with value. This step is the reason why num_heads is moved to batch dim, not len dim.
        query_state = torch.bmm(attn_weight, v) # `(bsz*self.num_heads, q_len, self.head_dim)`
        query_state = query_state.transpose(0, 1).contiguous().view(q_len, bsz, embed_dim)
        
        # combine heads
        query_state = self.combine_heads(query_state) # `(q_len, bsz, embed_dim)`
        query_state = query_state.transpose(0, 1).contiguous()
        
        outputs = (query_state,)
        if return_attn_weights:
            outputs += (attn_weight.view(bsz, self.num_heads, q_len, kv_len),)
        if current_keys_values is not None:
            outputs += (current_keys_values,)
            
        return outputs