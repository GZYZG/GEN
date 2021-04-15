import graph_edits
import pytorch_graph_edit_networks as gen
from config .config import config
import torch
import os

if __name__ == "__main__":
    model_name = "GEN_crossent_repeat1_aids.pth"
    # model hyperparameters
    num_layers = 2
    dim_hid = 64
    datasets = ['aids', 'aids_degree_rules', 'degree_rules']  # , 'edit_cycles', 'game_of_life']
    dim_ins = [32, 64, 32]
    nonlin = torch.nn.ReLU()
    model = gen.GEN(num_layers = num_layers, dim_in = dim_ins[0], dim_hid = dim_hid, nonlin = nonlin)
    model.load_state_dict(torch.load(config.model_name))
    A = []
    X = []
    # GEN 不需要先确定目标graph
    # GEN 的重点是对时间序列的预测，预测当前graph在下一个时刻的可能状态。GEN 不是针对一对图来给出二者之间的编辑操作序列，
    # 而是针对graph给出下一个时刻的预测，重点是 graph 的 time-series 的预测
    #
    delta_pred, Epsilon_pred = model(torch.tensor(A, dtype=torch.float), torch.tensor(X, dtype=torch.float))
