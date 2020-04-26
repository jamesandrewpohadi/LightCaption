import torch

def check_params(model_path):
    state_dict = torch.load(model_path)
    num_params = 0
    for key in state_dict:
        num_params += state_dict[key].numel()
    print('Num params:',num_params)
    return num_params

check_params('models/decoder.efficientnetb0-glove.200d-hidden512-connected_cell-5-3000.pth')