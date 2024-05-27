import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional

class FlashAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super(FlashAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads)

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_output, _ = self.attention(hidden_states, hidden_states, hidden_states)
        return attention_output

class LlamaEncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int) -> None:
        super(LlamaEncoderLayer, self).__init__()
        self.attention = FlashAttention(hidden_size, num_heads)
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: Tensor) -> Tensor:
        attention_output = self.attention(hidden_states)
        hidden_states = self.layer_norm(hidden_states + attention_output)
        intermediate_output = self.dense(hidden_states)
        intermediate_output = self.dropout(intermediate_output)
        hidden_states = self.layer_norm(hidden_states + intermediate_output)
        return hidden_states

class LlamaModel(nn.Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, intermediate_size: int, model_path: Optional[str] = None) -> None:
        super(LlamaModel, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["lm_head"]
        )
        self.model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quantization_config)
        self.model.half().to('cuda')
        self.layers = nn.ModuleList([
            LlamaEncoderLayer(hidden_size, num_heads, intermediate_size) for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(hidden_size, self.tokenizer.vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        inputs = self.tokenizer(input_ids, return_tensors="pt").input_ids.to('cuda')
        outputs = self.model(inputs)
        return outputs.logits

def run_inference(model: LlamaModel, text: str) -> Tensor:
    inputs = model.tokenizer(text, return_tensors="pt").input_ids.to('cuda')
    with torch.no_grad():
        outputs = model(inputs)
    return outputs

if __name__ == '__main__':
    model = LlamaModel(num_layers=12, hidden_size=768, num_heads=12, intermediate_size=3072, model_path="<downloaded model path from hf>")
    text = "Hello, world!"
    outputs = run_inference(model, text)
    print(outputs)
