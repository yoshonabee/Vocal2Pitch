def get_general_args(parser):
    parser.add_argument("train_json", type=str)
    parser.add_argument("val_json", type=str)
    parser.add_argument("--task", default="onset_offset_detection", choices=['onset_offset_detection'])
    parser.add_argument("--seed", type=float, default=39)
    parser.add_argument("-j", "--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser

def get_data_args(parser):
    group = parser.add_argument_group("data")

    return parser

def get_model_args(parser):
    group = parser.add_argument_group("model")

    group.add_argument("--in_channel", type=int, default=1)
    group.add_argument("--batchnorm", action='store_true')
    group.add_argument("--dropout", type=float, default=0)
    group.add_argument("--output_dim", type=int, default=1)

    return parser

def get_training_args(parser):
    group = parser.add_argument_group("training")
  
    group.add_argument("--batch_size", type=int, default=128) 
    group.add_argument("--lr", type=float, default=0.0001)
    group.add_argument("--epochs", type=int, default=100)
    group.add_argument("--criterion", type=str, default="bceloss", choices=['crossentropy', 'bceloss'])
    group.add_argument("--inbalance_ratio", type=float, default=0)
    group.add_argument("--valid_interval", type=int, default=1)
    group.add_argument("--name", type=str, default="default")
    group.add_argument("--weight_decay", type=float, default=0)
    group.add_argument("--checkpoint", type=int, default=5)

    return parser

def get_predicting_args(parser):

    group = parser.add_argument_group("predicting")
    group.add_argument("audio_list")
    group.add_argument("model_path")
    group.add_argument("--output_dir", type=str, default=".")
    group.add_argument("--seed", type=int, default=39)
    group.add_argument("--batch_size", type=int, default=512)
    group.add_argument("--device", type=str, default="cuda:0")

    return parser

def get_make_result_args(parser):
    parser.add_argument("pred_onset_list")
    parser.add_argument("data_dir")
    parser.add_argument("--output_dir", type=str, default=".")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--min_pitch", type=int, default=45)
    parser.add_argument("--max_pitch", type=int, default=80)
    parser.add_argument("--onset_thres", type=float, default=0.02)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--crepe", action="store_true", help="use crepe's pitch tracking estimation")
    parser.add_argument("--crepe_confidence_thres", type=float, default=0.4)
    parser.add_argument("-j", "--num_workers", type=int, default=4)

    return parser

def get_evaluating_args(parser):
    parser.add_argument("predict_json")
    parser.add_argument("data_dir")

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
