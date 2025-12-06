# tools/confmat_dlf.py
# 用法示例：
#   python tools/confmat_dlf.py --dataset mosi --ckpt ./pt/DLF_mosi.pth --acc_mode acc7
#   python tools/confmat_dlf.py --dataset mosei --ckpt ./pt/DLF_mosei.pth --acc_mode acc2 --normalize true
#
# 说明：
# - 自动从项目里加载 DLF 模型与测试集，推理得到 y_true / y_pred，然后绘制混淆矩阵。
# - 模型若输出回归分数（[-3,3]），会按 --acc_mode 将其离散为 Acc2/Acc7 类别；
#   若输出分类 logits（形状 [B, K]），则直接 argmax 作为预测类别。
# - 默认保存两张图：counts（原始计数）和 norm_true（行归一化），文件在 ./viz/ 下。

import os
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ======== 复用项目内部模块 ========
from config import get_config_regression
from data_loader import MMDataLoader
from utils import assign_gpu, setup_seed
from trains.singleTask.model.DLF import DLF


LABEL_NAMES_ACC7 = ["HN", "N", "WN", "NT", "WP", "P", "HP"]  # -3..3
LABEL_NAMES_ACC2 = ["NEG", "POS"]                            # <=0 , >0


def build_args(dataset: str, gpu: str):
    model_name = "DLF"
    dataset_name = "mosi"
    args = get_config_regression(model_name,dataset_name)           # 项目自带
    args['model_name'] = 'DLF'
    args['task'] = 'regression'              # DLF 默认是回归任务；如果你做分类，这里不影响推理分支
    args['dataset_name'] = dataset.lower()   # 'mosi' / 'mosei'
    args['device'] = assign_gpu(gpu)
    args['is_train'] = False
    args['feature_T'] = ""
    args['feature_A'] = ""
    args['feature_V'] = ""
    setup_seed(42)
    return args


@torch.no_grad()
def move_to_device(batch, device):
    """把 batch 里的张量递归搬到 device 上."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return [move_to_device(v, device) for v in batch]
    return batch


def _first_tensor_in(container):
    """从可能是张量/字典/列表的结构里，取第一个张量返回。"""
    if isinstance(container, torch.Tensor):
        return container
    if isinstance(container, dict):
        for v in container.values():
            t = _first_tensor_in(v)
            if isinstance(t, torch.Tensor):
                return t
    if isinstance(container, (list, tuple)):
        for v in container:
            t = _first_tensor_in(v)
            if isinstance(t, torch.Tensor):
                return t
    return None


def extract_label_from_batch(batch):
    """兼容不同命名：label / labels / y / target / sentiment 等."""
    candidates = ['labels', 'label', 'y', 'target', 'sentiment', 'gt']
    for k in candidates:
        if k in batch:
            t = _first_tensor_in(batch[k])
            if t is not None:
                return t
    # 如果没有明确键，退回到 batch 的第一个张量
    t = _first_tensor_in(batch)
    if t is None:
        raise RuntimeError("在 batch 中未找到标签张量，请检查数据加载器的键名（如 'labels'/'label' 等）")
    return t


def model_forward(model, batch):
    """尽量兼容不同前向签名：model(**batch) / model(batch) / model(t, a, v)."""
    # 仅保留张量型输入，避免把 label 传进模型
    inputs = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            # 过滤常见的标签键
            if k.lower() in ['labels', 'label', 'y', 'target', 'sentiment', 'gt']:
                continue
            inputs[k] = v

    try:
        out = model(**inputs)          # 大多数项目走这里
        return out
    except TypeError:
        try:
            out = model(inputs)        # 有的模型直接吃 dict
            return out
        except Exception:
            # 兜底：按常见模态顺序传
            keys = [k for k in ['text', 'language', 'words',
                                'audio', 'acoustic',
                                'vision', 'video', 'visual'] if k in inputs]
            out = model(*[inputs[k] for k in keys])
            return out


def to_class_from_regression(scores: np.ndarray, mode: str):
    """把回归分数（[-3,3]）映射为 Acc2/Acc7 类别索引."""
    scores = np.clip(scores, -3.0, 3.0)
    if mode == 'acc2':
        # >0 为 POS(1)；<=0 为 NEG(0)
        return (scores > 0).astype(int)
    elif mode == 'acc7':
        # 四舍五入到最近的整数，再平移到 0..6
        cls = np.rint(scores).astype(int)            # -3..3 的整数
        cls = np.clip(cls, -3, 3)
        return cls + 3                               # -3->0, ..., 3->6
    else:
        raise ValueError("mode 必须是 acc2 或 acc7")


def pick_pred_tensor(out):
    """从模型前向输出中挑出用于评估的张量：
       - 若是 dict，优先 'logits'/'pred'/'output'/'score'；
       - 若张量形状为 [B,K]，认为是分类 logits；若 [B] 或 [B,1]，认为是回归分数。
    """
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, dict):
        for k in ['logits', 'pred', 'output', 'score', 'y_hat']:
            if k in out and isinstance(out[k], torch.Tensor):
                return out[k]
        # 否则取第一个张量
        t = _first_tensor_in(out)
        if t is not None:
            return t
    if isinstance(out, (list, tuple)):
        for v in out:
            t = pick_pred_tensor(v)
            if isinstance(t, torch.Tensor):
                return t
    raise RuntimeError("无法从模型输出中找到预测张量，请检查 forward 的返回结构。")


@torch.no_grad()
def run_inference_get_labels_and_preds(model, loader, device, acc_mode: str):
    y_true_all, y_pred_all = [], []
    model.eval()
    for batch in loader:
        batch = move_to_device(batch, device)
        # 取真实标签（连续值，通常在 [-3,3]）
        y_true_t = extract_label_from_batch(batch).float().view(-1)

        # 前向
        out = model_forward(model, batch)
        y_pred_t = pick_pred_tensor(out)

        # 分类 or 回归两种情况：
        if y_pred_t.dim() == 2 and y_pred_t.size(1) >= 2:
            # 分类：取 argmax 得类别索引
            pred_cls = torch.argmax(y_pred_t, dim=1).view(-1)
            y_pred_all.append(pred_cls.cpu().numpy())
            # 若是真实标签是回归分数，则把真实也映射到对应类别空间
            if y_true_t.min() < 0 or y_true_t.max() > 1 or y_true_t.dtype.is_floating_point:
                if y_pred_t.size(1) == 2:
                    y_true_all.append(to_class_from_regression(y_true_t.cpu().numpy(), 'acc2'))
                else:
                    y_true_all.append(to_class_from_regression(y_true_t.cpu().numpy(), 'acc7'))
            else:
                y_true_all.append(y_true_t.long().cpu().numpy())
        else:
            # 回归：先拿连续分数，再按 acc_mode 离散
            scores = y_pred_t.view(-1).cpu().numpy()
            y_pred_all.append(to_class_from_regression(scores, acc_mode))
            y_true_all.append(to_class_from_regression(y_true_t.cpu().numpy(), acc_mode))

    y_true = np.concatenate(y_true_all, axis=0)
    y_pred = np.concatenate(y_pred_all, axis=0)
    return y_true, y_pred


def plot_and_save_confmat(y_true, y_pred, labels, out_dir: Path, prefix: str, normalize=None):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)), normalize=normalize)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=180)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           ylabel='Label (True)',
           xlabel='Predicted')
    # 网格 & 数字
    thresh = cm.max() / 2. if cm.size > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]:.2f}" if normalize else f"{int(cm[i, j])}"
            ax.text(j, i, text,
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)
    ax.set_title(f"Confusion Matrix ({'normalized' if normalize else 'counts'})")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{prefix}_{'norm' if normalize else 'counts'}.png"
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=['mosi', 'mosei'], help="数据集名称")
    parser.add_argument("--ckpt", type=str, required=True, help="模型 .pth 路径")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--acc_mode", type=str, default="acc7", choices=['acc2','acc7'],
                        help="回归输出时的离散方式")
    parser.add_argument("--normalize", type=str, default="none",
                        choices=['none','true','pred','all'],
                        help="混淆矩阵归一化方式（绘图时用）")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    cfg = build_args(args.dataset, args.gpu)
    device = cfg['device']

    # ====== DataLoader（项目内封装）======
    loaders = MMDataLoader(cfg,4)
    if isinstance(loaders, (list, tuple)) and len(loaders) >= 3:
        train_loader, valid_loader, test_loader = loaders[:3]
    else:
        # 如果项目版本不同，尝试命名函数
        test_loader = MMDataLoader(cfg,4)

    # ====== 模型与权重 ======
    model = DLF(cfg).to(device)
    state = torch.load(args.ckpt, map_location=device)
    # 兼容只存 state_dict 或 整包
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    model.load_state_dict(state, strict=False)
    model.eval()

    # ====== 推理并离散为类别 ======
    y_true, y_pred = run_inference_get_labels_and_preds(model, test_loader, device, args.acc_mode)

    # ====== 指标报告 ======
    if args.acc_mode == 'acc2' or (y_pred.max() <= 1):
        label_names = LABEL_NAMES_ACC2
    else:
        label_names = LABEL_NAMES_ACC7

    print(classification_report(y_true, y_pred, target_names=label_names, digits=4))

    # ====== 绘图保存 ======
    out_dir = Path("./viz/confmat")
    prefix = f"{args.dataset}_{args.acc_mode}"
    # 原始计数
    plot_and_save_confmat(y_true, y_pred, label_names, out_dir, prefix, normalize=None)
    # 按真实标签归一化（常用）
    norm_map = {'none': None, 'true': 'true', 'pred': 'pred', 'all': 'all'}
    if args.normalize != 'none':
        plot_and_save_confmat(y_true, y_pred, label_names, out_dir, prefix, normalize=norm_map[args.normalize])


if __name__ == "__main__":
    main()
