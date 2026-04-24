import torch
import torch.nn as nn


class Item2VecModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        # 中心 Embedding
        self.center_embeddings = nn.Embedding(vocab_size, embed_dim)
        # 初始化权重
        nn.init.normal_(self.center_embeddings.weight, mean=0.0, std=0.01)

    def forward(self, center_idx, context_idx, neg_idx):
        """
        前向计算：正样本得分 + 负样本得分
        """
        # [B, D]
        center_emb = self.center_embeddings(center_idx)
        context_emb = self.center_embeddings(context_idx)
        # [B * neg_count, D] -> [B, neg_count, D]
        neg_emb = self.center_embeddings(neg_idx).view(
            center_idx.size(0), -1, self.embed_dim
        )

        # 正样本相似度
        pos_score = torch.sum(center_emb * context_emb, dim=1)  # [B]

        # 负样本相似度
        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(-1)).squeeze(-1)  # [B, neg_count]

        return pos_score, neg_score
