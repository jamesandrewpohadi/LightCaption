import torch

def check_params(model_path):
    state_dict = torch.load(model_path)
    num_params = 0
    for key in state_dict:
        num_params += state_dict[key].numel()
    print('Num params:',num_params)
    return num_params

check_params('models/encoder.efficientnetb1-hidden256-connected_cell_hidden-vocab10-scale_down3-5-4000.pth')