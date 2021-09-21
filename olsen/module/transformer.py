import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel


class BertSentenceEncoder(nn.Module):
    def __init__(
            self,
            embedding_method: str,
            transformer: str,
    ):
        super(BertSentenceEncoder, self).__init__()
        self.embedding_method = embedding_method
        self.model_name_or_path = transformer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModel.from_pretrained(self.model_name_or_path)
        self.dimension = self.get_dimension()
        self.attention = nn.MultiheadAttention(embed_dim=self.model.config.hidden_size, batch_first=True, num_heads=8)

    def forward(self, inputs) -> torch.Tensor:
        model_output = self.model(**inputs)
        output = model_output.last_hidden_state

        # batch_size, bert_hidden_dimension
        if self.embedding_method == "cls":
            cls_mask = inputs.input_ids.eq(self.tokenizer.cls_token_id)
            sentence_encoding = output[cls_mask, :].view(output.size(0), -1, output.size(-1)).squeeze(1)
        elif self.embedding_method == "eos":
            eos_mask = inputs.input_ids.eq(self.tokenizer.eos_token_id)
            sentence_encoding = output[eos_mask, :].view(output.size(0), -1, output.size(-1)).squeeze(1)
        elif self.embedding_method == "mean_pool":
            attention_mask = inputs.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).float()
            sum_embeddings = torch.sum(output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            sentence_encoding = sum_embeddings / sum_mask
        elif self.embedding_method == "max_pool":
            attention_mask = inputs.attention_mask
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).float()
            output[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            sentence_encoding = torch.max(output, dim=1)
        elif self.embedding_method == "attn_pool_cls":
            attention_mask = inputs.attention_mask
            cls_mask = inputs.input_ids.eq(self.tokenizer.cls_token_id)
            query = output[cls_mask, :].view(output.size(0), -1, output.size(-1))
            key = output
            attn = self.attention(query=query, key=key, value=key, key_padding_mask=attention_mask)[0]
            sentence_encoding = attn.squeeze(1)
        elif self.embedding_method == "attn_pool_eos":
            attention_mask = inputs.attention_mask
            eos_mask = inputs.input_ids.eq(self.tokenizer.eos_token_id)
            query = output[eos_mask, :].view(output.size(0), -1, output.size(-1))
            key = output
            attn = self.attention(query=query, key=key, value=key, key_padding_mask=attention_mask)[0]
            sentence_encoding = attn.squeeze(1)
        else:
            raise ValueError(f"The embedding type {self.embedding_method} does not exist")
        return sentence_encoding

    def get_dimension(self) -> int:
        return self.model.config.hidden_size
