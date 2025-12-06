import json
import logging
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ..utils import MetricsTop, dict_to_str
from .HingeLoss import HingeLoss


logger = logging.getLogger('MMSA')

class MSE(nn.Module):
    def __init__(self):
        super(MSE, self).__init__()

    def forward(self, pred, real):
        diffs = torch.add(real, -pred)
        n = torch.numel(diffs.data)
        mse = torch.sum(diffs.pow(2)) / n
        return mse

class DLF():
    def __init__(self, args):
        self.args = args
        self.criterion = nn.L1Loss()           
        self.cosine = nn.CosineEmbeddingLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)
        self.MSE = MSE()
        self.sim_loss = HingeLoss()

    def do_train(self, model, dataloader, return_epoch_results=False):

        # 0: DLF model
        params = model[0].parameters() 

        optimizer = optim.Adam(params, lr=self.args.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=self.args.patience)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        net = []
        net_DLF = model[0]
        net.append(net_DLF)    
        model = net
        
        while True:
            epochs += 1
            y_pred, y_true = [], []
            for mod in model:
                mod.train()
              

            train_loss = 0.0
            all_features = []  # 存前4个batch的特征
            save_batch_count = 20  # 仅保存前4个batch
            all_labels = []

            all_c_l = []
            all_c_v = []
            all_c_a = []
            all_s_l = []
            all_s_v = []
            all_s_a = []
            left_epochs = self.args.update_epochs
            with tqdm(dataloader['train']) as td:
                for j, batch_data in enumerate(td):

                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)


                   
                    output = model[0](text, audio, vision) 

                    # task loss
                    loss_task_all = self.criterion(output['output_logit'], labels)
                    
                    loss_task_l_hetero = self.criterion(output['logits_l_hetero'], labels)  
                    loss_task_v_hetero = self.criterion(output['logits_v_hetero'], labels)
                    loss_task_a_hetero = self.criterion(output['logits_a_hetero'], labels)
                    loss_task_c = self.criterion(output['logits_c'], labels)
                    
                    # total MSA loss L_msa
                    loss_task = 1* (1 * loss_task_all + 1*loss_task_c  + 3 * loss_task_l_hetero + 1*loss_task_v_hetero + 1*loss_task_a_hetero)
                    
                    # reconstruction loss L_r
                    loss_recon_l = self.MSE(output['recon_l'], output['origin_l'])
                    loss_recon_v = self.MSE(output['recon_v'], output['origin_v'])
                    loss_recon_a = self.MSE(output['recon_a'], output['origin_a'])
                    loss_recon = loss_recon_l + loss_recon_v + loss_recon_a

                    # specific loss L_s 
                    loss_sl_slr = self.MSE(output['s_l'].permute(1, 2, 0), output['s_l_r'])
                    loss_sv_slv = self.MSE(output['s_v'].permute(1, 2, 0), output['s_v_r'])
                    loss_sa_sla = self.MSE(output['s_a'].permute(1, 2, 0), output['s_a_r'])
                    loss_s_sr = loss_sl_slr + loss_sv_slv + loss_sa_sla

                    # ort loss L_o
                    if self.args.dataset_name == 'mosi':
                        num = 50
                    elif self.args.dataset_name == 'mosei':
                        num = 10

                    cosine_similarity_s_c_l = self.cosine(output['s_l'].reshape(-1, num), output['c_l'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_v = self.cosine(output['s_v'].reshape(-1, num), output['c_v'].reshape(-1, num), torch.tensor([-1]).cuda())
                    cosine_similarity_s_c_a = self.cosine(output['s_a'].reshape(-1, num), output['c_a'].reshape(-1, num), torch.tensor([-1]).cuda())
                    
                    loss_ort = cosine_similarity_s_c_l + cosine_similarity_s_c_v + cosine_similarity_s_c_a

                    # triplet margin loss L_m
                    c_l, c_v, c_a = output['c_l_sim'], output['c_v_sim'], output['c_a_sim']
                    ids, feats = [], []
                    for i in range(labels.size(0)):
                        feats.append(c_l[i].view(1, -1))
                        feats.append(c_v[i].view(1, -1))
                        feats.append(c_a[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                        ids.append(labels[i].view(1, -1))
                    feats = torch.cat(feats, dim=0)
                    ids = torch.cat(ids, dim=0)
                    loss_sim = self.sim_loss(ids, feats)

                    #overall loss L_DLF
                    combined_loss = loss_task + (loss_s_sr    + loss_recon   + (loss_sim   +loss_ort ) * 0.1) * 0.1  +  0.02 * output['moe_aux']
                
                    combined_loss.backward()

                    # # ===== 收集前4个batch的特征与标签 =====
                    # if j < save_batch_count:
                    #     features = output['last_hs_proj'].detach().cpu().numpy().tolist()
                    #     labels_np = labels.detach().cpu().numpy().flatten().tolist()
                    #
                    #     all_features.extend(features)
                    #     all_labels.extend(labels_np)
                    #
                    # # ===== 到第4个batch时保存一次并停止 =====
                    # if j + 1 ==           :
                    #     save_data = {
                    #         "features": all_features,  # 特征向量
                    #         "labels": all_labels  # 浮点真实标签
                    #     }
                    #     save_path = f"last_hs_proj_first_{save_batch_count}_batches.json"
                    #     with open(save_path, "w", encoding="utf-8") as f:
                    #         json.dump(save_data, f, ensure_ascii=False, indent=2)
                    #     print(f"✅ 已保存前 {save_batch_count} 个 batch 的特征与标签到 {save_path}")


                    # # ===== 收集前4个batch的特征与标签 =====
                    if j < save_batch_count:
                        c_l_mean = output['c_l'].permute(1, 0, 2).mean(dim=1)  # [B, D]
                        c_v_mean = output['c_v'].permute(1, 0, 2).mean(dim=1)
                        c_a_mean = output['c_a'].permute(1, 0, 2).mean(dim=1)

                        s_l_mean = output['s_l'].permute(1, 0, 2).mean(dim=1)
                        s_v_mean = output['s_v'].permute(1, 0, 2).mean(dim=1)
                        s_a_mean = output['s_a'].permute(1, 0, 2).mean(dim=1)

                        # 如果你想确认维度，可以打印一次
                        # print("c_l:", output['c_l'].shape, "->", c_l_mean.shape)

                        # 转为 Python 列表并累加
                        all_c_l.extend(c_l_mean.detach().cpu().numpy().tolist())
                        all_c_v.extend(c_v_mean.detach().cpu().numpy().tolist())
                        all_c_a.extend(c_a_mean.detach().cpu().numpy().tolist())

                        all_s_l.extend(s_l_mean.detach().cpu().numpy().tolist())
                        all_s_v.extend(s_v_mean.detach().cpu().numpy().tolist())
                        all_s_a.extend(s_a_mean.detach().cpu().numpy().tolist())

                    # ===== 到第4个batch时保存一次并停止 =====
                    if j + 1 == save_batch_count:
                        save_data = {
                            "c_l": all_c_l,  # 共享特征：语言部分
                            "c_v": all_c_v,  # 共享特征：视觉部分
                            "c_a": all_c_a,  # 共享特征：音频部分
                            "s_l": all_s_l,  # 语言特定特征
                            "s_v": all_s_v,  # 视觉特定特征
                            "s_a": all_s_a,  # 音频特定特征
                            "labels": all_labels  # 浮点真实标签
                        }
                        save_path = f"features_and_labels_first_{save_batch_count}_batches.json"
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(save_data, f, ensure_ascii=False, indent=2)

                        print(f"✅ 已保存前 {save_batch_count} 个 batch 的共享特征、模态特定特征与标签到 {save_path}")


                    if self.args.grad_clip != -1.0:
                        params = list(model[0].parameters())  
                                 
                        nn.utils.clip_grad_value_(params, self.args.grad_clip)

                    train_loss += combined_loss.item()
                    

                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())
                    if not left_epochs:
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            

            train_loss = train_loss / len(dataloader['train'])
            pred, true = torch.cat(y_pred), torch.cat(y_true)


            train_results = self.metrics(pred, true)
            logger.info(
                f">> Epoch: {epochs} "
                f"TRAIN -({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> total_loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            # validation
            val_results = self.do_test(model[0], dataloader['valid'], mode="VAL")
            test_results = self.do_test(model[0], dataloader['test'], mode="TEST")
            cur_valid = val_results[self.args.KeyEval]
            scheduler.step(val_results['Loss'])
            # save each epoch model
            torch.save(model[0].state_dict(), './pt/' + str(self.args.dataset_name) + '_' + str(epochs) + '.pth')
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                model_save_path = './pt/DLF' + str(self.args.dataset_name)+'.pth'
                torch.save(model[0].state_dict(), model_save_path)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):

        model.eval()
        y_pred, y_true = [], []

        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        all_features = []  # 存前4个batch的特征
        save_batch_count = 20  # 仅保存前4个batch
        all_labels = []
        all_c_l = []
        all_c_v = []
        all_c_a = []
        all_s_l = []
        all_s_v = []
        all_s_a = []
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for i, batch_data in enumerate(td):
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    labels = batch_data['labels']['M'].to(self.args.device)
                    labels = labels.view(-1, 1)

                    output = model(text, audio, vision)

                    loss = self.criterion(output['output_logit'], labels)
                    eval_loss += loss.item()
                    y_pred.append(output['output_logit'].cpu())
                    y_true.append(labels.cpu())

                    # # # ===== 收集前4个batch的特征与标签 =====
                    # if i < save_batch_count:
                    #
                    #     c_l = output['c_l'].detach().cpu().numpy().tolist()
                    #     c_v = output['c_v'].detach().cpu().numpy().tolist()
                    #     c_a = output['c_a'].detach().cpu().numpy().tolist()
                    #     s_l = output['s_l'].detach().cpu().numpy().tolist()
                    #     s_v = output['s_v'].detach().cpu().numpy().tolist()
                    #     s_a = output['s_a'].detach().cpu().numpy().tolist()
                    #
                    #     all_c_l.extend(c_l)
                    #     all_c_v.extend(c_v)
                    #     all_c_a.extend(c_a)
                    #     all_s_l.extend(s_l)
                    #     all_s_v.extend(s_v)
                    #     all_s_a.extend(s_a)
                    #
                    #
                    # # ===== 到第4个batch时保存一次并停止 =====
                    # if i + 1 == save_batch_count:
                    #     save_data = {
                    #         "c_l": all_c_l,  # 共享特征：语言部分
                    #         "c_v": all_c_v,  # 共享特征：视觉部分
                    #         "c_a": all_c_a,  # 共享特征：音频部分
                    #         "s_l": all_s_l,  # 语言特定特征
                    #         "s_v": all_s_v,  # 视觉特定特征
                    #         "s_a": all_s_a,  # 音频特定特征
                    #         "labels": all_labels  # 浮点真实标签
                    #     }
                    #     save_path = f"features_and_labels_first_{save_batch_count}_batches.json"
                    #     with open(save_path, "w", encoding="utf-8") as f:
                    #         json.dump(save_data, f, ensure_ascii=False, indent=2)
                    #
                    #     print(f"✅ 已保存前 {save_batch_count} 个 batch 的共享特征、模态特定特征与标签到 {save_path}")

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)

        test_preds = pred.view(-1).cpu().detach().numpy()
        test_truth = true.view(-1).cpu().detach().numpy()

        test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
        test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
        output_file = 'test_preds_vs_truth_a7.txt'
        with open(output_file, 'w') as f:
            for x, y in zip(test_preds_a7, test_truth_a7):
                f.write(f"{x:.6f}\t{y:.6f}\n")  # 使用制表符分隔，保留6位小数

        eval_results = self.metrics(pred, true)
        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results