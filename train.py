"""
Training script for DLF
"""
from run import DLF_run
if __name__ == '__main__':

    DLF_run(model_name='DLF', dataset_name='mosi', is_tune=False, seeds=[], model_save_dir="./pt",
             res_save_dir="./result", log_dir="./log", mode='train', is_training=True)
