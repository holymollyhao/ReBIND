"""
    This script is used to train a Conformer model for conformer prediction.
"""

import warnings
import torch
from datasets import DatasetDict
from dataclasses import dataclass

from transformers import EvalPrediction
from transformers import TrainingArguments, Trainer
from transformers import HfArgumentParser
from transformers.trainer_utils import get_last_checkpoint

from models import REBINDConfig, REBIND, Collator
import numpy as np
import random, os
import wandb

device = "cuda" if torch.cuda.is_available() else "cpu"
warnings.simplefilter("ignore", UserWarning)


def init_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class ScriptArguments:
    preprocessed_dataset_path: str
    dataset: str
    num_layers: int
    split: str


def compute_metrics(eval_pred: EvalPrediction):
    preds, _ = eval_pred
    mae, mse, rmsd, *_ = preds
    mae, mse, rmsd = mae.mean().item(), mse.mean().item(), rmsd.mean().item()

    return {"mae": mae, "mse": mse, "rmsd": rmsd}


def main():
    parse = HfArgumentParser(dataclass_types=[ScriptArguments, TrainingArguments])
    script_args, training_args = parse.parse_args_into_dataclasses()
    print(script_args)
    print(training_args)
    
    init_seed(training_args.seed)


    dataset = DatasetDict.load_from_disk(script_args.preprocessed_dataset_path)
    
    train_set, eval_set, test_set = (
        dataset["train"],
        dataset["validation"],
        dataset["test"],
    )
    
    collate_func = Collator()

    config = REBINDConfig(
        # encoder config
        n_encode_layers=script_args.num_layers,
        
        # decoder config
        n_decode_layers=script_args.num_layers,
        embed_style="atom_type_ids",
        
        # model config
        atom_vocab_size=513,  # 512 + 1 for padding (shifted all input ids by 1)
        d_embed=512,
        pre_ln=False,  # layer norm before residual, else after residual, not Pre-LN and not Post-LN.
        d_q=512,
        d_k=512,
        d_v=512,
        d_model=512,
        n_head=8,
        qkv_bias=True,
        attn_drop=0.,
        norm_drop=0.,
        ffn_drop=0.,
        dropout=0.,
        d_ffn=1024,
    )

    model = REBIND(config)
    print(model.config)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_func,
        train_dataset=train_set,
        eval_dataset=eval_set,
        compute_metrics=compute_metrics,
    )
    
    # training
    if training_args.do_train:
        resume_from_checkpoint = training_args.resume_from_checkpoint
        try:
            resume_f = eval(resume_from_checkpoint)
            if not resume_f:
                train_result = trainer.train()
            else:
                last_checkpoint = get_last_checkpoint(training_args.output_dir)
                train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        except Exception:
            train_result = trainer.train()
        trainer.log_metrics("train", train_result.metrics)

    # testing
    if training_args.do_eval:
        # TODO: if training_args.do_train is False, model used to test is not the best!
        test_metrics = trainer.evaluate(eval_dataset=test_set)
        trainer.log_metrics("test", test_metrics)




if __name__ == "__main__":
    main()
