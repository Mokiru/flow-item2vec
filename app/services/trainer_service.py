import gc
import random
import time
from collections import defaultdict
from datetime import datetime
from itertools import permutations
from typing import List, Dict

import numpy as np
import torch
from torch import optim
import torch.nn.functional as F

from nn_models.item2vec import Item2VecModel


class Item2VecTrainer:
    def __init__(
            self,
            vector_store,
            datasource,
            dim=128,
            lr=0.01,
            neg_count=5,
            batch_size=2000,
            grad_clip_norm=1.0,
            max_items_per_order=50,
            max_safe_pairs=20000
    ):
        self.vector_store = vector_store
        self.datasource = datasource
        self.dim = dim
        self.lr = lr
        self.neg_count = neg_count
        self.batch_size = batch_size
        self.grad_clip_norm = grad_clip_norm
        self.max_items_per_order = max_items_per_order
        self.max_safe_pairs = max_safe_pairs

        self.device = torch.device("cpu")
        torch.set_flush_denormal(True)
        torch.set_num_threads(1)

        self.model = None
        self.optimizer = None
        self.unique_bcs = []
        self.bc_to_idx = {}

    def _build_model_from_vectors(self, unique_bcs, train_date: str):
        """
        使用SGD优化器，初始化Embedding时，对于非当日数据加入时间权重噪声；初始化全新商品
        :param unique_bcs: 当前批订单中唯一barcode列表
        :param train_date: 训练日期
        :return: None
        """
        vocab_size = len(unique_bcs)
        self.unique_bcs = unique_bcs
        self.bc_to_idx = {bc: i for i, bc in enumerate(unique_bcs)}

        self.model = Item2VecModel(vocab_size, self.dim).to(self.device)
        stored_data = self.vector_store.get_vectors(unique_bcs)

        weight_np = np.zeros((vocab_size, self.dim), dtype=np.float32)
        today = datetime.strptime(train_date, '%Y-%m-%d').date()

        for i, bc in enumerate(unique_bcs):
            data = stored_data.get(bc, {})
            w = data.get('w', None)
            last_day = data.get('last_train_day', '')

            if w is not None and last_day == train_date:
                weight_np[i] = w
            elif w is not None:
                try:
                    last_day_date = datetime.strptime(last_day, '%Y-%m-%d').date()
                    days_gap = (today - last_day_date).days
                except Exception as ignore:
                    days_gap = 1
                noise_scale = min(0.0005 * days_gap, 0.02)
                noise = np.random.normal(0, noise_scale, self.dim).astype(np.float32)
                weight_np[i] = w + noise
            else:
                raw_vec = np.random.randn(self.dim).astype(np.float32)  # 随机方向
                vec_norm = np.linalg.norm(raw_vec)  # 模长
                if vec_norm > 1e-6:
                    weight_np[i] = (raw_vec / vec_norm) * 0.1
                else:
                    weight_np[i] = np.random.normal(0, 0.01, self.dim).astype(np.float32)

        with torch.no_grad():
            self.model.center_embeddings.weight.copy_(torch.from_numpy(weight_np))
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def _generate_positive_pairs(self, orders):
        """
        创建商品对
        :param orders:
        :return: 中心商品barcode，上下文商品barcode，商品对
        """
        pairs = []
        for order in orders:
            order_list = list(order)
            if len(order_list) > self.max_items_per_order:
                random.shuffle(order_list)
                order_list = order_list[:self.max_items_per_order]
            if len(order_list) < 2:
                continue
            pairs.extend(permutations(order_list, 2))
        if not pairs:
            return None, None, None
        center_bcs, context_bcs = zip(*pairs)
        return center_bcs, context_bcs, pairs

    def _sample_negatives(self, batch_size, neg_count, idx_pool, sampling_probs, ln_q_arr):
        """
        全局无差别负采样
        :param batch_size: 正样本对的总数量 (即 len(center_bcs))
        :param neg_count: 每个正样本需要的负样本数量
        :param idx_pool: 候选词索引数组 (通常为 np.arange(len(unique_bcs)))
        :param sampling_probs: 对应 idx_pool 的采样概率数组 (必须和为 1)
        :param ln_q_arr: 对应 idx_pool 的 ln_q 修正值数组
        :return: (neg_idx_list, neg_lnq_list) 扁平化的一维 numpy array
        """
        total_neg_need = batch_size * neg_count
        neg_idx_flat = np.random.choice(idx_pool, size=total_neg_need, p=sampling_probs)
        neg_idx_result = neg_idx_flat.reshape(batch_size, neg_count)
        neg_lnq_result = ln_q_arr[neg_idx_result]
        return neg_idx_result.flatten(), neg_lnq_result.flatten()

    def _build_negative_sampler_from_data(self, unique_bcs, sampling_data):
        """
        负采样基础数据构建
        :param unique_bcs: 唯一barcode列表
        :param sampling_data:  采样数据
        :return:
        """
        prob_list = []
        ln_q_dict = {}
        for bc in unique_bcs:
            data = sampling_data.get(bc, {})
            prob = max(float(data.get("samplingProb", 1e-8)), 1e-8)
            prob_list.append(prob)
            ln_q_dict[bc] = float(data.get("lnProbCorrection", -18.42068))

        prob_arr = np.array(prob_list, dtype=np.float64)
        prob_arr /= prob_arr.sum()
        idx_pool = np.arange(len(unique_bcs), dtype=np.int64)
        return idx_pool, prob_arr, ln_q_dict

    def train_with_data(self, orders: List[List[str]], sampling_data: Dict, train_date) -> int:
        """
        执行单批次训练
        返回 1 成功，0 无效数据
        """
        if not orders:
            return 0

        # 生成正样本
        center_bcs, context_bcs, pairs = self._generate_positive_pairs(orders)
        if not pairs:
            return 0

        unique_bcs = list(set(center_bcs) | set(context_bcs))
        self._build_model_from_vectors(unique_bcs, train_date)

        idx_pool, sampling_probs, ln_q_dict = self._build_negative_sampler_from_data(unique_bcs,
                                                                                     sampling_data)  # 采样数据与损失修正常数

        # 构建 ln_q_arr 损失修正常数
        ln_q_arr = np.array([ln_q_dict.get(bc, -18.42068) for bc in unique_bcs], dtype=np.float32)

        # 转换为 Tensor
        center_idx = torch.tensor([self.bc_to_idx[b] for b in center_bcs], dtype=torch.int64, device=self.device)
        context_idx = torch.tensor([self.bc_to_idx[b] for b in context_bcs], dtype=torch.int64, device=self.device)
        batch_size = center_idx.size(0)
        neg_idx_list, neg_lnq_list = self._sample_negatives(
            batch_size=batch_size,
            neg_count=self.neg_count,
            idx_pool=idx_pool,
            sampling_probs=sampling_probs,
            ln_q_arr=ln_q_arr
        )

        # 负采样
        neg_idx = torch.from_numpy(np.array(neg_idx_list, dtype=np.int64))
        neg_lnq_t = torch.from_numpy(
            np.array(neg_lnq_list, dtype=np.float32).reshape(batch_size, self.neg_count)
        )

        # 一批单epoch训练
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        i = 0
        ignore_size = self.max_safe_pairs * 0.3

        while i < batch_size:
            step = self.max_safe_pairs
            next_i = i + step # 下一片起始
            if batch_size - next_i < ignore_size and next_i < batch_size: # 最后两片可以合二为一
                step = batch_size - i
            end_idx = i + step
            c_batch = center_idx[i:end_idx]
            ctx_batch = context_idx[i:end_idx]
            neg_start = i * self.neg_count
            neg_end = end_idx * self.neg_count
            neg_batch = neg_idx[neg_start:neg_end]
            lnq_batch = neg_lnq_t[i:end_idx]
            pos_score, neg_score = self.model(c_batch, ctx_batch, neg_batch)
            neg_score_corrected = neg_score - lnq_batch.detach()
            loss = (F.softplus(-pos_score) + F.softplus(neg_score_corrected).sum(1)).mean()
            total_loss += loss.item() * (end_idx - i)
            loss.backward()
            i = end_idx

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        print(f"[商品对: {batch_size:,}] Loss: {total_loss / batch_size:.4f}")

        # 5. 回写与清理
        self._flush_embeddings_to_vector_store(train_date)
        self._clear()
        return 1

    def _flush_embeddings_to_vector_store(self, train_date):
        """极简回写：只存向量 W 和日期，彻底告别 m/v"""
        self.model.eval()
        with torch.no_grad():
            weights = self.model.center_embeddings.weight.cpu().numpy()

        w_flat = weights.tolist()

        update_entities = []
        for i, bc in enumerate(self.unique_bcs):
            update_entities.append([bc, w_flat[i], train_date])

        self.vector_store.upsert_new_vectors(update_entities)

    def _clear(self):
        """极简清理"""
        if self.model:
            self.model.zero_grad(set_to_none=True)
            for p in self.model.parameters():
                p.detach_()
            del self.model
            self.model = None

        if self.optimizer:
            del self.optimizer
            self.optimizer = None

        gc.collect()
