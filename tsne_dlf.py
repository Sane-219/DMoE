# tools/tsne_dlf.py
# 运行示例:
#   python tools/tsne_dlf.py --dataset mosei --gpu 0
#   python tools/tsne_dlf.py --dataset mosi  --gpu 0

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ======== 复用项目里的构建/数据/训练器 ========
from config import get_config_regression
from data_loader import MMDataLoader
from trains import ATIO
from utils import assign_gpu, setup_seed
from trains.singleTask.model import DLF

def build_args(model_name: str, dataset_name: str, gpu_ids):
    """
    基于 run.py 的做法构建 args（测试模式）:
      - config/config.json
      - ./pt/DLF{dataset}.pth
    """
    config_file = "./config/config.json"
    # if not config_file.is_file():
    #     raise FileNotFoundError(f"Config not found: {config_file}")
    args = get_config_regression(model_name.upper(), dataset_name.lower(), config_file)
    args.is_training = False
    args.mode = "test"                          # 与 run.py 一致
    args['device'] = assign_gpu(gpu_ids)        # GPU 选择
    args['train_mode'] = 'regression'           # 与项目一致

    args['feature_T'] = args.get('feature_T', "")
    args['feature_A'] = args.get('feature_A', "")
    args['feature_V'] = args.get('feature_V', "")
    # 模型保存路径（虽然本脚本只测试，但字段保留一致）
    if 'model_save_path' not in args:
        args['model_save_path'] = Path.home() / "MMSA" / "saved_models" / f"{args['model_name']}-{args['dataset_name']}.pth"
    return args

def find_last_linear(model: nn.Module) -> nn.Linear:
    last_linear = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last_linear = m
    if last_linear is None:
        raise RuntimeError("未找到分类头(nn.Linear)。如你的分类头不是 Linear，请手动改成相应模块。")
    return last_linear

@torch.no_grad()
@torch.no_grad()
def collect_embeddings_with_test_loop(model, test_loader, args_device):
    """
    1) 给最后一个 Linear 注册 forward-hook，抓其“输入”作为 z（分类器之前的联合表示）
    2) 走标准测试流程跑一遍前向（hook 自动收集 z）
    3) 再遍历一次 test_loader，使用健壮的递归函数从 batch 中提取标签张量
    """
    model.eval().to(args_device)

    # ---- 1) 找分类头并挂 hook
    last_linear = find_last_linear(model)
    buf_z = []
    def _hook(module, inputs, output):
        # 分类器的“输入”就是我们要的联合表示 z
        buf_z.append(inputs[0].detach().cpu())
    handle = last_linear.register_forward_hook(_hook)

    # ---- 2) 标准测试流程（与你项目一致）
    trainer = ATIO().getTrain(args)  # 注意：使用全局 args
    _ = trainer.do_test(model, test_loader, mode="TEST")

    handle.remove()

    # ---- 3) 收集标签（健壮提取）
    def _to_tensor(x):
        """把任意对象转成 1D 张量（尽量保留顺序）"""
        if torch.is_tensor(x):
            t = x
        elif isinstance(x, np.ndarray):
            t = torch.from_numpy(x)
        elif isinstance(x, (list, tuple)):
            # 例如一批标量
            try:
                t = torch.tensor(x)
            except Exception:
                return None
        else:
            return None
        # 压到 1D
        if t.dim() > 1:
            t = t.view(t.size(0), -1)[:, 0]
        return t

    def extract_label_from(obj):
        """递归在 dict / list 里寻找合适的标签张量"""
        # 直接就是张量/数组/列表
        t = _to_tensor(obj)
        if t is not None:
            return t

        # dict：优先常见键名；否则遍历所有值
        if isinstance(obj, dict):
            # 常见键名优先
            for k in ["label", "labels", "regression_label", "regression_labels", "y", "target", "M", "m"]:
                if k in obj:
                    t = extract_label_from(obj[k])
                    if t is not None:
                        return t
            # 兜底：遍历所有 value
            for v in obj.values():
                t = extract_label_from(v)
                if t is not None:
                    return t

        # 序列：逐个尝试
        if isinstance(obj, (list, tuple)):
            for v in obj:
                t = extract_label_from(v)
                if t is not None:
                    return t

        return None

    ys = []
    for batch in test_loader:
        # 直接从原始 batch 递归提取（不需要搬到 device）
        y = None
        # 常见顶层键先试
        if isinstance(batch, dict):
            for k in ["label", "labels", "regression_label", "regression_labels", "y", "target"]:
                if k in batch:
                    y = extract_label_from(batch[k])
                    if y is not None:
                        break
        # 还没拿到就全面递归
        if y is None:
            y = extract_label_from(batch)

        if y is None:
            raise RuntimeError(f"无法在 batch 中提取标签。可打印 batch.keys() 检查实际结构。")

        ys.append(y.detach().cpu())

    Z = torch.cat(buf_z, dim=0).numpy()          # [N, D]
    y = torch.cat(ys, dim=0).numpy().astype(float)
    return Z, y


def map_to_three_classes(y_cont_or_7cls: np.ndarray):
    """
    将连续[-3,3]或7类{-3,-2,...,+3}统一映射到 {neg=0, neu=1, pos=2}
    阈值 ±0.2 常用、样本更均衡。
    """
    pos = y_cont_or_7cls >  0.2
    neg = y_cont_or_7cls < -0.2
    y3 = np.full_like(y_cont_or_7cls, 1, dtype=int)
    y3[pos] = 2
    y3[neg] = 0
    return y3

def run_tsne(Z: np.ndarray, seed: int = 42):
    Zs = StandardScaler().fit_transform(Z)
    n = len(Zs)
    perplexity = min(30, max(5, (n - 1) // 3))
    tsne = TSNE(
        n_components=2,
        perplexity=40,  # 30~50 对几千样本较稳
        learning_rate='auto',  # 或 200~800
        early_exaggeration=24,  # 12~36 可试
        n_iter=3000,
        init='pca',
        metric='cosine',  # 语音/文本常更稳
        angle=0.5,
        random_state=42,
    )
    return tsne.fit_transform(Zs)  # [N, 2]

def plot_tsne(X2: np.ndarray, y3: np.ndarray, title: str, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.6, 4.8), dpi=180)
    markers = {0: "x", 1: "s", 2: "o"}
    names   = {0: "Negative", 1: "Neutral", 2: "Positive"}
    for c in [2, 1, 0]:
        idx = (y3 == c)
        plt.scatter(X2[idx, 0], X2[idx, 1],
                    s=18, marker=markers[c], alpha=0.85, label=names[c])
    plt.xticks([]); plt.yticks([])
    plt.title(title)
    plt.legend(frameon=True, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    print(f"[t-SNE] Saved -> {out_path}")

def load_model_and_loader(args, num_workers=1):
    """
    与 run.py 一致的构建方式：
      - DLF(args)
      - MMDataLoader(args, num_workers)
      - ./pt/DLF{dataset}.pth 权重
    """
    dataloader = MMDataLoader(args, num_workers)
    model = DLF.DLF(args).to(args['device'])
    ckpt = r"D:/苏志炎/DLF-main - 副本/DLF-main-1/DLF-main-4/pt/mosei_3.pth"
    # if not ckpt.is_file():
    #     raise FileNotFoundError(f"未找到权重: {ckpt}（请按 run.py 约定命名放在 ./pt/ 目录下）")
    sd = torch.load(str(ckpt), map_location=args['device'])
    model.load_state_dict(sd, strict=False)  # 与 run.py 测试逻辑一致
    return model, dataloader['test']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mosei", choices=["mosei", "mosi"],
                        help="选择数据集（与权重命名 ./pt/DLF{dataset}.pth 对应）")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, 如 0")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--title", type=str, default=None, help="图标题（默认自动生成）")
    parser.add_argument("--out", type=str, default="figs/tsne_dlf.png", help="输出路径")
    args_cli = parser.parse_args()

    # ========== 构建项目 args ==========
    global args
    setup_seed(args_cli.seed)
    args = build_args("DLF", args_cli.dataset, gpu_ids=[args_cli.gpu])

    # ========== 模型 + 测试集 ==========
    model, test_loader = load_model_and_loader(args, num_workers=args_cli.num_workers)

    # ========== 收集联合表示 z & 标签 ==========
    Z, y = collect_embeddings_with_test_loop(model, test_loader, args['device'])

    # ========== t-SNE ==========
    X2 = run_tsne(Z, seed=args_cli.seed)
    y3 = map_to_three_classes(y)

    # ========== 作图 ==========
    title = args_cli.title or f"DLF (+MOEFFN) classifier-pre {{A,V,L}} on {args_cli.dataset.upper()}"
    plot_tsne(X2, y3, title, Path(args_cli.out))

if __name__ == "__main__":
    main()
