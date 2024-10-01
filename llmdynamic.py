import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer
from typing import List, Tuple, Optional

class DynamicAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.lazy_threshold = config.lazy_threshold

        self.c_attn = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.c_proj = nn.Linear(config.hidden_size, config.hidden_size)

        self.pending_updates_K: List[List[Tuple[int, int, float]]] = [[] for _ in range(self.num_heads)]
        self.pending_updates_V: List[List[Tuple[int, int, float]]] = [[] for _ in range(self.num_heads)]
        self.num_updates = 0

    def _split_heads(self, x, num_heads, head_dim):
        new_shape = x.size()[:-1] + (num_heads, head_dim)
        x = x.view(*new_shape)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_shape)

    def apply_lazy_updates(self, K, V):
        for head in range(self.num_heads):
            for i, j, delta in self.pending_updates_K[head]:
                K[:, head, i, j] += delta
            for i, j, delta in self.pending_updates_V[head]:
                V[:, head, i, j] += delta

        self.pending_updates_K = [[] for _ in range(self.num_heads)]
        self.pending_updates_V = [[] for _ in range(self.num_heads)]
        self.num_updates = 0
        return K, V

    def lazy_update_K(self, updates: List[Tuple[int, int, int, float]]):
        for head, i, j, delta in updates:
            self.pending_updates_K[head].append((i, j, delta))
        self.num_updates += len(updates)

    def lazy_update_V(self, updates: List[Tuple[int, int, int, float]]):
        for head, i, j, delta in updates:
            self.pending_updates_V[head].append((i, j, delta))
        self.num_updates += len(updates)

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None, use_cache=False, output_attentions=False):
        B, T, C = x.size()

        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)

        q = self._split_heads(q, self.num_heads, self.head_dim)
        k = self._split_heads(k, self.num_heads, self.head_dim)
        v = self._split_heads(v, self.num_heads, self.head_dim)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        if self.num_updates >= self.lazy_threshold:
            k, v = self.apply_lazy_updates(k, v)

        present = torch.stack((k, v)) if use_cache else None

        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, v)
        attn_output = self._merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class DynamicAttentionGPT2(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        for i, block in enumerate(self.transformer.h):
            block.attn = DynamicAttention(config)

    def lazy_update_K(self, layer: int, updates: List[Tuple[int, int, int, float]]):
        self.transformer.h[layer].attn.lazy_update_K(updates)

    def lazy_update_V(self, layer: int, updates: List[Tuple[int, int, int, float]]):
        self.transformer.h[layer].attn.lazy_update_V(updates)

# Example usage
def main():
    config = GPT2Config.from_pretrained('gpt2-medium')
    config.lazy_threshold = 10  # Set the lazy update threshold

    model = DynamicAttentionGPT2(config)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')

    
    # Generate text (this will trigger the application of lazy updates)
    input_text = "How are you today ?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    attention_mask = torch.ones_like(input_ids)

    # Set pad_token_id to eos_token_id if it's None
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id

    output = model.generate(
    input_ids,
    attention_mask=attention_mask,
    pad_token_id=model.config.pad_token_id,
    eos_token_id=model.config.eos_token_id,
    max_length=200,  # Increase max length
    num_return_sequences=1,
    no_repeat_ngram_size=3,  # Adjust no repeat n-gram size
    do_sample=True,
    top_k=100,  # Increase top-k
    top_p=0.98,  # Increase top-p
    temperature=0.7,  # Adjust temperature
    num_beams=5,  # Enable beam search
    early_stopping=True,  # Stop generation when all beams have reached the end token
    )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()
