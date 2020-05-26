def get_general_args(parser):
    parser.add_argument("train_json", type=str)
    parser.add_argument("val_json", type=str)
    parser.add_argument("--task", default="onset_offset_detection", choices=['onset_offset_detection'])
    parser.add_argument("--seed", type=float, default=39)

    return parser

def get_data_args(parser):
    group = parser.add_argument_group("data")

    group.add_argument("--thres", type=int, default=0.3)
    group.add_argument("--segment_length", type=float, default=4)
    group.add_argument("--sr", type=int, default=16000)

    return parser

def get_model_args(parser):
    group = parser.add_argument_group("model")

    group.add_argument("--model_config", type=str, default="config/model.json")
    group.add_argument("--in_channel", type=int, default=1)
    group.add_argument("--batchnorm", action='store_true')
    group.add_argument("--dropout", type=float, default=0)
    group.add_argument("--output_dim", type=int, default=1)

    return parser

def get_training_args(parser):
    group = parser.add_argument_group("training")

    group.add_argument("--device", type=str, default="cuda:0")  
    group.add_argument("--batch_size", type=int, default=128) 
    group.add_argument("--lr", type=float, default=0.001)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--criterion", type=str, default="crossentropy", choices=['crossentropy', 'bceloss'])
    group.add_argument("--inbalance_ratio", type=float, default=0)
    group.add_argument("--valid_interval", type=int, default=1)
    group.add_argument("--name", type=str, default="default")
    group.add_argument("--weight_decay", type=float, default=0)
    group.add_argument("--checkpoint", type=int, default=5)

    return parser

def set_seed(seed):
    import random
    import torch
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
