from argparse import ArgumentParser

import json
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from utils import get_general_args, get_data_args, get_model_args, get_training_args, set_seed
from data import Dataset
from model import CNN_RNN
from criterion import ResampleCriterion

from pytorch_trainer import Trainer
from pytorch_trainer.metrics import Accuracy, Precision, Recall, F1

torch.set_num_threads(4)

def main(args):
    if args.task == "onset_offset_detection":
        model_config = json.load(open(args.model_config, 'r'))
        model = CNN_RNN(
            layers_config=model_config,
            segment_length=4
        )

        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        train_dataset = Dataset(
            args.train_json,
            thres=args.thres,
            segment_length=args.segment_length,
        )

        val_dataset = Dataset(
            args.val_json,
            thres=args.thres,
            segment_length=args.segment_length,
        )

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

        trainer = Trainer(
            model,
            train_dataloader,
            val_dataloader,
            device=torch.device(args.device),
            loss_fn=ResampleCriterion(args.inbalance_ratio, args.criterion),
            metrics=[Accuracy(args.criterion == "bceloss"), Precision(args.criterion == "bceloss"), Recall(args.criterion == "bceloss"), F1(args.criterion == "bceloss")],
            lr=args.lr,
            optimizer=optimizer,
            epochs=args.epochs,
            checkpoint=args.checkpoint,
            valid_interval=args.valid_interval,
            save_dir=f"models/{args.name}",
            log_dir=f"runs/{args.name}"
        )

        trainer.train()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = get_general_args(parser)
    parser = get_data_args(parser)
    parser = get_model_args(parser)
    parser = get_training_args(parser)

    args = parser.parse_args()
    set_seed(args.seed)

    print(args)

    main(args)

