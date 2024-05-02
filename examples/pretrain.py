import argparse
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import numpy as np
from pathlib import Path
from time import time
from transformers import get_cosine_schedule_with_warmup
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from types import FunctionType
from typing import Tuple
import wandb

from flygpt import DataCollator
from flygpt.model import TransformerModel
from flygpt.tokenizer.gene_tokenizer import GeneVocab, get_default_gene_vocab
from flygpt.utils import set_seed
from flygpt.loss import (
    masked_mse_loss,
    masked_relative_error,
    criterion_neg_log_bernoulli,
)


hyperparameter_defaults = dict(
    project_name="flyGPT",
    dataset_name="flycorpus",
    seed=42,
    do_train=True,
    bin_expression=False,  # Binning expression values by cell
    bin_size=None,  # Number of expression bins within sample
    max_seq_len=1536,  #  max sequence length
    training_tasks="both",  # gen/pcpt/both
    generative_training=True,  # set to false if pcpt only
    mask_ratio=0.4,  # ignored if training-tasks set to gen/both
    epochs=30,
    nbins=None,  # 51,
    domain_spec_batchnorm=False,
    GEP=True,  # Gene expression
    GEPC=True,  # Masked value prediction for cell embedding
    CLS=False,  # Cell type classification objective (default True)
    CCE=True,  # Contrastive cell embedding objective
    ecs_threshold=0.8,  # Elastic cell similarity objective, [0, 1]. 0 => dis
    dab_weight=1.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    save_eval_interval=5,
    log_interval=100,
    checkpoint_steps=1000,  # Save model every n steps. if "epoch" after epoch
    explicit_zero_prob=False,
    fast_transformer=True,
    pre_norm=False,
    fp16=True,  # Automatic Mixed Precision
    eval_ratio=0.03,  # Evaluation ratio for pretraining
    trunc_by_sampling=False,  # Truncate seq by sampling instead of slice
    mask_value=-1,
    pad_value=-2,
    pad_token="<pad>",
    num_proc=10,
    # Scheduler
    scheduler_interval=100,  # interval iterations for updating the LR
    scheduler_factor=0.99,  # ratio of epochs for learning rate schedule
    warmup_ratio_or_step=0.1,  # ratio out of the total training steps.
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="flyGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
set_seed(config.seed)
criterion_cls = nn.CrossEntropyLoss()


def update_config():
    """Update config for coherence"""
    if config.training_tasks in ["gen", "both"]:
        config.update({
            "mask_ratio": [0.25, 0.5, 0.75],
            "generative_training": True,
        }, allow_val_change=True)
    else:
        config.update({
            "generative_training": False,
        }, allow_val_change=True)

    print(config)



def parse_args():
    parser = argparse.ArgumentParser(
        description="Pretrain flyGPT model"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        required=True,
        help="Path to processed dataset. See `preprocessed",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default="./data.scb",
        help="Directory to save scBank data, by default will make a "
        "directory named data.scb in the current directory",
    )
    # vocabulary
    parser.add_argument(
        "-v",
        "--vocab-file",
        type=str,
        default=None,
        help="File containing the gene vocabulary, default to None. "
        "If None, will use the default gene vocabulary from flyGPT, "
        "which use FlyBase gene symbols.",
    )
    return parser.parse_args()


def get_data_loader(
        dataset: Dataset, 
        collator: DataCollator,
        shuffle: bool = True,
    ) -> DataLoader:
    """Get data loader for the given dataset"""

    sampler = DistributedSampler(dataset, shuffle=True)

    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        collate_fn=collator,
        drop_last=False,
        num_workers=1,  # min(config.num_proc, config.batch_size),
        pin_memory=True,
        prefetch_factor=4,
    )


def load_and_tokenize_data(
        args: argparse.Namespace, 
        vocab: GeneVocab
    ) -> Tuple[DataLoader, DataLoader]:
    """
    Load and tokenize data
    

    Returns
    -------
    Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]
    Training and evaluation dataloaders
    
    """

    collator = DataCollator(
        do_padding=True if config.max_seq_len is not None else False,
        pad_token_id=vocab[config.pad_token],
        pad_value=config.pad_value,
        do_mlm=True,
        do_binning=config.bin_expression,
        mlm_probability=config.mask_ratio,
        mask_value=config.mask_value,
        max_length=config.max_seq_len,
        sampling=config.trunc_by_sampling,
        data_style=config.training_tasks,
    )

    dataset = load_from_disk(args.dataset)
    print(f"Loaded dataset with {len(dataset)} samples")

    # Convert to PyTorch tensor
    dataset = dataset.with_format("torch")

    train, eval = dataset.train_test_split(
        test_size=config.eval_ratio,
        shuffle=True
    ).values()

    train_loader = get_data_loader(train, collator)
    eval_loader = get_data_loader(eval, collator, shuffle=False)

    return train_loader, eval_loader


def init_model(vocab: GeneVocab) -> nn.Module:
    """Initialize the model with the given vocabulary"""

    return TransformerModel(
        ntoken=len(vocab),
        d_model=config.layer_size,
        nhead=config.nhead,
        d_hid=config.layer_size,
        nlayers=config.nlayers,
        nlayers_cls=3,  # default
        n_cls=1,  # default
        dropout=config.dropout,
        pad_token=config.pad_token,
        pad_value=config.pad_value,
        do_mvc=config.GEPC,
        do_dab=False,  # default
        use_batch_labels=False,  # default
        num_batch_labels=None,  # default
        input_emb_style="continuous",  # default
        n_input_bins=config.nbins,
        explicit_zero_prob=config.explicit_zero_prob,
        use_generative_training=config.generative_training,
        use_fast_transformer=config.fast_transformer,
        fast_transformer_backend="flash",
        pre_norm=config.pre_norm,
    )


def init_scheduler(optimizer, train_loader: DataLoader):
    if config.warmup_ratio_or_step > 0:
        total_num_batches = len(train_loader) * config.epochs
        warmup_steps = (
            int(total_num_batches * config.warmup_ratio_or_step)
            if config.warmup_ratio_or_step < 1
            else int(config.warmup_ratio_or_step)
        )
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_num_batches,
            last_epoch=-1,
        )
    else:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            config.scheduler_interval, 
            gamma=config.scheduler_factor
        )


def update_log_loss(
        total_loss: dict, 
        loss: dict, 
        batch_idx: int,
        num_batches: int,
        global_step: int,
        batch_time: float,
        scheduler: torch.optim.lr_scheduler.StepLR,
        accelerator: Accelerator
    ) -> float:
    """Update total loss with the current batch loss"""
    
    def gather(tensor: torch.Tensor):
        """Gather tensor across all devices"""
        t = accelerator.gather_for_metrics(tensor)
        print(t)
        return t

    # Update total loss tensors
    for key, value in loss.items():
        total_loss[key] += value

    # If batch is a multiple of log interval, log the loss
    if batch_idx % config.log_interval == 0 and batch_idx > 0:
        # Log training loss
        loss_log = { 
            f"train_{key}": (gather(value) / config.log_interval).item()
            for key, value in total_loss.items()
        }
        ms_per_batch = (time() - batch_time) * 1000 / config.log_interval
        accelerator.log({
            **loss_log,
            "lr": scheduler.get_last_lr()[0],
            "train_batch": f"{batch_idx} / {num_batches}",
            "ms/batch": f"{ms_per_batch:5.2f}"
        }, step=global_step)

        # Reset total loss
        total_loss = { key: torch.tensor(0.0, device=accelerator.device)
                       for key in total_loss }

    return batch_time


def train_generative(
        model: nn.Module,
        batch: dict,
        vocab: GeneVocab,
        global_step: int,
    ) -> Tuple[torch.Tensor, dict]:
    """"""
    pcpt_gene = batch["pcpt_gene"]
    pcpt_expr = batch["pcpt_expr"]
    pcpt_key_padding_mask = pcpt_gene.eq(vocab[config.pad_token])
    gen_gene = batch["gen_gene"]
    # TODO: return target values to compute global mre
    gen_expr_target = target_values = batch["gen_expr_target"]
    gen_key_padding_mask = gen_gene.eq(vocab[config.pad_token])

    outputs = model(
        pcpt_gene,
        pcpt_expr,
        pcpt_key_padding_mask,
        gen_gene,
        gen_key_padding_mask,
        CLS=config.CLS,
        MVC=config.GEP,
        generative_training=True,
    )
    gen_expr_preds = output_values = outputs["gen_preds"]

    positions_to_match = ~gen_key_padding_mask

    # Keep track of losses
    losses = {}

    loss = loss_mse = masked_mse_loss(
        gen_expr_preds, 
        gen_expr_target, 
        positions_to_match
    )
    losses["mse"] = loss_mse


    if config.GEP:
        loss_mvc = masked_mse_loss(
            outputs["mvc_output"][:, pcpt_gene.shape[1] :],
            gen_expr_target,
            positions_to_match,
        )
        loss += loss_mvc
        losses["mvc"] = loss_mvc

    # Generative pretraining when enough iterations
    if global_step > 1000:
        previous_cell_embs = outputs["cell_emb"].detach()
        preds = model(
            pcpt_gene,
            pcpt_expr,
            pcpt_key_padding_mask,
            gen_gene,
            gen_key_padding_mask,
            CLS=False,
            MVC=False,
            input_cell_emb=previous_cell_embs,
            generative_training=True,
        )["gen_preds"]
        loss_gen = masked_mse_loss(preds, gen_expr_target, positions_to_match)
        loss += loss_gen
        losses["gen"] = loss_gen

    return output_values, { "loss": loss, **losses }


def train_masked(
        model: nn.Module,
        batch: dict,
        vocab: GeneVocab,
        _: int,  # global_step unused
    ) -> Tuple[torch.Tensor, dict]:
    """"""
    input_gene_ids = batch["gene"]
    input_values = batch["masked_expr"]
    target_values = batch["expr"]
    src_key_padding_mask = input_gene_ids.eq(vocab[config.pad_token])
    
    outputs = model(
        input_gene_ids,
        input_values,
        src_key_padding_mask=src_key_padding_mask,
        CLS=config.CLS,
        CCE=config.CCE,  # TODO: move these flags to model's attributes
        MVC=config.GEP,
        generative_training=False,
    )

    output_values = outputs["mlm_output"]
    
    # Positions to predict
    positions_to_match = input_values.eq(config.mask_value)

    # Keep track of losses
    losses = {}

    loss = loss_mse = masked_mse_loss(
        output_values,
        target_values,
        positions_to_match
    )
    losses["mse"] = loss_mse

    if config.CLS:
        target_labels = batch["celltypes"]
        loss_cls = criterion_cls(outputs["cls_output"], target_labels)
        loss += loss_cls
        losses["cls"] = loss_cls
    if config.CCE:
        loss_cce = 10 * outputs["loss_cce"]
        loss += loss_cce
        losses["cce"] = loss_cce
    if config.GEP:
        loss_mvc = masked_mse_loss(
            outputs["mvc_output"], target_values, positions_to_match
        )
        loss += loss_mvc
        losses["mvc"] = loss_mvc


    return output_values, { "loss": loss, **losses }


def train(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocab: GeneVocab,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.StepLR,
        accelerator: Accelerator,
        train_handler: FunctionType,
        epoch: int,
        output_dir: Path,
    ):
    """Train the model for a single epoch"""

    # Initialize training
    model.train()

    # Keep track of total losses
    total_loss = {
        "loss": torch.tensor(0.0, device=accelerator.device),
        "mse": torch.tensor(0.0, device=accelerator.device),
        "cls": torch.tensor(0.0, device=accelerator.device),
        "gen": torch.tensor(0.0, device=accelerator.device),
        "mvc": torch.tensor(0.0, device=accelerator.device),
    }
    total_error = torch.tensor(0.0, device=accelerator.device)
    batch_time = time()
    
    n_batches = len(dataloader)

    # Training loop
    for batch_idx, batch in enumerate(dataloader):
        global_step = epoch * n_batches + batch_idx

        # Zero the gradients before running the backward pass
        # i.e. if you do not want to accumulate gradients for multiple 
        # iterations. Do not place between backward() and optimizer.step()
        optimizer.zero_grad()
        
        # Perform forward pass and compute loss (generative or masked)
        outputs, loss = train_handler(model, batch, vocab, global_step)

        # Backward pass and optimization
        accelerator.backward(loss["loss"])

        # Perform gradient clipping to avoid exploding gradients
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        # Update total loss and log to wandb
        batch_time = update_log_loss(
            total_loss=total_loss, 
            loss=loss, 
            batch_idx=batch_idx, 
            num_batches=n_batches,
            global_step=global_step,
            batch_time=batch_time, 
            scheduler=scheduler,
            accelerator=accelerator
        )

        # Save the model, optimizer, lr_scheduler, and seed states
        # Will contain files: "pytorch_model.bin", "optimizer.bin", 
        # "scheduler.bin", and "random_states.pkl"
        # If mixed precision was used, will also save a "scalar.bin" file
        if (isinstance(config.checkpoint_steps, int) and 
            global_step % config.checkpoint_steps == 0):
                step_dir = f"epoch_{epoch}/step_{global_step}"
                accelerator.save_state(str(output_dir / step_dir))
            
    if config.checkpoint_steps == "epoch":
        accelerator.save_state(str(output_dir / f"epoch_{epoch}"))


def evaluate_generative(
        model: nn.Module,
        batch: dict,
        vocab: GeneVocab,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """"""

    pcpt_gene = batch["pcpt_gene"]
    pcpt_expr = batch["pcpt_expr"]
    pcpt_key_padding_mask = pcpt_gene.eq(vocab[config.pad_token])
    gen_gene = batch["gen_gene"]
    target_values = batch["gen_expr_target"]
    gen_key_padding_mask = gen_gene.eq(vocab[config.pad_token])

    outputs = model(
        pcpt_gene,
        pcpt_expr,
        pcpt_key_padding_mask,
        gen_gene,
        gen_key_padding_mask,
        CLS=False,
        MVC=False,
        generative_training=True,
    )
    output_values = outputs["gen_preds"]

    positions_to_match = ~gen_key_padding_mask

    return output_values, target_values, positions_to_match


def evaluate_masked(
        model: nn.Module,
        batch: dict,
        vocab: GeneVocab,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """"""
    return


def evaluate(
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        vocab: GeneVocab,
        accelerator: Accelerator,
        eval_handler: FunctionType,
        epoch: int,
    ) -> None:
    """Evaluate the model for a single epoch"""
    model.eval()

    total_mse = torch.tensor(0.0, device=accelerator.device)
    total_mre = torch.tensor(0.0, device=accelerator.device)

    with torch.no_grad():
        for batch in dataloader:

            output_values, target_values, positions_to_match = eval_handler(
                model=model, 
                batch=batch, 
                vocab=vocab, 
            )
            
            # Gather output and target values across all devices
            output_values, target_values = accelerator.gather_for_metrics(
                (output_values, target_values)
            )

            mse = masked_mse_loss(
                output_values, 
                target_values, 
                positions_to_match
            )
            mre = masked_relative_error(
                output_values, 
                target_values, 
                positions_to_match
            )

            total_mse += mse
            total_mre += mre

        # Log total mse and mre for the epoch
        accelerator.log({ 
            "valid/mse": total_mse.item() / len(dataloader), 
            "valid/mre": total_mre.item() / len(dataloader) 
        }, epoch)


def main():

    update_config()

    args = parse_args()

    if args.vocab_file is None:
        vocab = get_default_gene_vocab()
    else:
        # TODO: GeneVocab should have a default built in
        vocab = GeneVocab.from_file(args.vocab_file)

    # TODO: Special tokens should already be in vocab
    special_tokens = [config.pad_token, "<cls>", "<eoc>"]

    # Initialize Accelerate Accelerator (multi-gpu training)
    # Track model metrics with wandb
    # Do before loading data to inititialize GPU distribution properly
    accelerator = Accelerator(log_with="wandb", 
                              mixed_precision="fp16" if config.fp16 else None)

    # Load data
    train_dataloader, eval_dataloader = load_and_tokenize_data(args, vocab)

    # Initialize model
    model = init_model(vocab)

    # Initialize Adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.lr, 
        # eps=1e-4 if config.amp else 1e-8
    )
    # Initialize learning rate scheduler
    scheduler = init_scheduler(optimizer, train_dataloader)

    # Prepare components for acceleration
    (   model, 
        optimizer, 
        train_dataloader, 
        eval_dataloader, 
        scheduler) = accelerator.prepare(
        model, 
        optimizer, 
        train_dataloader,
        eval_dataloader, 
        scheduler
    )

    accelerator.init_trackers(config.project_name)

    for epoch in range(config.epochs):
        train_handler = (train_generative if config.generative_training 
                                          else train_masked)
        # Perform model training
        train(model=model, 
              vocab=vocab, 
              dataloader=train_dataloader,
              optimizer=optimizer,
              scheduler=scheduler,
              accelerator=accelerator,
              train_handler=train_handler,
              epoch=epoch,
              output_dir=Path(args.output_dir) / "training")

        eval_handler = (evaluate_generative if config.generative_training 
                                            else evaluate_masked)
        # Perform model evaluation
        evaluate(model=model, 
                 vocab=vocab, 
                 dataloader=eval_dataloader,
                 accelerator=accelerator,
                 eval_handler=eval_handler,
                 epoch=epoch)

    accelerator.end_training()


if __name__ == "__main__":
    main()
    run.finish()
    print("Training completed successfully!")
