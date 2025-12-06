import torch
from torch import nn
import torch.nn.functional as F

class MoEFFN(nn.Module):
    """
    Top-2 Mixture-of-Experts FFN，兼容输入 [T,B,D]
    - num_experts: 专家数
    - top_k: 路由的专家数（建议 2）
    - capacity_factor: 容量系数；容量 = ceil(N * top_k / E * factor)
    - expert_dropout: 训练时按专家级随机丢弃，稳路由
    前向会把负载均衡正则写到 self.last_aux_loss（eval 时为 0）
    """
    def __init__(self, d_model, d_hidden,
                 num_experts=8, top_k=2, capacity_factor=1.25,
                 noisy_gating=True, dropout=0.1, expert_dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden = d_hidden
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.noisy_gating = noisy_gating
        self.expert_dropout = expert_dropout

        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_hidden),
                nn.ReLU(),          # 与原 FFN 保持一致（原实现用 ReLU）
                nn.Dropout(dropout),
                nn.Linear(d_hidden, d_model),
            ) for _ in range(num_experts)
        ])
        for e in self.experts:
            nn.init.xavier_uniform_(e[0].weight); nn.init.constant_(e[0].bias, 0.)
            nn.init.xavier_uniform_(e[3].weight); nn.init.constant_(e[3].bias, 0.)

        self.last_aux_loss = None

    def forward(self, x_tbd: torch.Tensor):
        # x_tbd: [T,B,D]
        T, B, D = x_tbd.shape
        x = x_tbd.reshape(T*B, D)                 # [N,D]
        N = x.size(0)

        logits = self.router(x)                   # [N,E]
        if self.training and self.noisy_gating:
            logits = logits + torch.randn_like(logits) * 1e-2

        topv, topi = torch.topk(logits, k=self.top_k, dim=-1)   # [N,k], [N,k]
        gate = F.softmax(topv, dim=-1)                           # [N,k]
        E, k = self.num_experts, self.top_k

        with torch.no_grad():
            expected = N * k / float(E)
            capacity = int(torch.clamp(torch.tensor(expected * self.capacity_factor).ceil(), min=1).item())

        y = x.new_zeros(x.shape)                  # [N,D]
        usage = x.new_zeros(E)
        mean_gate = x.new_zeros(E)

        if self.training and self.expert_dropout > 0:
            drop_mask = (torch.rand(E, device=x.device) > self.expert_dropout).float()
        else:
            drop_mask = torch.ones(E, device=x.device)

        for e in range(E):
            # 聚合同一 token 命中的多个位置（k<=2）
            gate_e = torch.zeros(N, device=x.device, dtype=x.dtype)
            for j in range(k):
                gate_e += gate[:, j] * (topi[:, j] == e).float()

            n_tokens = int((gate_e > 0).sum().item())
            if n_tokens == 0 or drop_mask[e] == 0.0:
                continue

            if n_tokens > capacity:
                topg, idx = torch.topk(gate_e, k=capacity, dim=0)
                chosen = idx
                gate_use = topg
            else:
                chosen = (gate_e > 0).nonzero(as_tuple=False).squeeze(-1)
                gate_use = gate_e[chosen]

            out_e = self.experts[e](x[chosen])              # [m,D]
            y[chosen] += out_e * gate_use.unsqueeze(-1) * drop_mask[e]

            usage[e] = float(chosen.numel()) / max(1, N)
            mean_gate[e] = gate_use.mean() if gate_use.numel() > 0 else 0.0

        y = y.view(T, B, D)

        # 负载均衡正则（鼓励使用率与门值分布接近均匀）
        target = 1.0 / E
        aux = ((usage - target) ** 2 + (mean_gate - target) ** 2).mean()
        self.last_aux_loss = aux if self.training else y.new_zeros(())

        return y