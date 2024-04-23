import argparse
from accelerate import Accelerator
from datasets import Dataset, load_from_disk
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Tuple

from flygpt import DataCollator
from flygpt.model import TransformerModel
from flygpt.tokenizer.gene_tokenizer import GeneVocab, get_default_gene_vocab
from flygpt.utils import set_seed
from flygpt.loss import (
    masked_mse_loss,
    criterion_neg_log_bernoulli,
)


hyperparameter_defaults = dict(
    seed=42,
    dataset_name="flycorpus",
    do_train=True,
    bin_size=None,  # Number of expression bins within sample
    max_seq_len=1536,  #  max sequence length
    training_task="both",  # gen/pcpt/both
    generative_training=True,  # set to false if pcpt only
    mask_ratio=0.4,  # ignored if training-tasks set to gen/both
    epochs=30,
    nbins=None,  # 51,
    domain_spec_batchnorm=False,
    GEP=True,  # Gene expression
    GEPC=True,  # Masked value prediction for cell embedding
    ecs_threshold=0.8,  # Elastic cell similarity objective, [0, 1]. 0 => dis
    dab_weight=1.0,
    lr=1e-4,
    batch_size=32,
    layer_size=128,
    nlayers=4,
    nhead=4,
    dropout=0.2,
    schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
    save_eval_interval=5,
    log_interval=100,
    explicit_zero_prob=False,
    fast_transformer=True,
    pre_norm=False,
    amp=True,  # Automatic Mixed Precision
    eval_ratio=0.03,  # Evaluation ratio for pretraining
    trunc_by_sampling=False,  # Truncate seq by sampling instead of slice
    pad_value=-1,
    mask_value=-2,
    num_proc=10,
    pad_token="<pad>",
)
run = wandb.init(
    config=hyperparameter_defaults,
    project="flyGPT",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)
config = wandb.config
print(config)
set_seed(config.seed)


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
        type=str,
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
        num_workers=min(config.num_proc, config.batch_size),
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
        do_binning=True if args.input_style == "binned" else False,
        mlm_probability=config.mask_ratio,
        mask_value=config.mask_value,
        max_length=config.max_seq_len,
        sampling=config.trunc_by_sampling,
        data_style=config.training_tasks,
    )

    dataset = load_from_disk(args.dataset)
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
        vocab=vocab,
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


def compute_loss(
        outputs: dict, 
        target_values: torch.Tensor, 
        masked_positions: torch.Tensor, 
        celltype_labels: torch.Tensor, 
        batch_labels: torch.Tensor
    ):
    """Compute loss and metrics to log for the model"""
    loss = 0.0
    metrics_to_log = {}
    # Gene expression prediction loss (MSE)
    if config.GEP:
        loss_gep = masked_mse_loss(
            outputs["mlm_output"], target_values, masked_positions
        )
        loss += loss_gep
        metrics_to_log = {"train/gep": loss_gep.item()}

    # Non zero log prob loss (only when predicting gene expression)
    if config.GEP and config.explicit_zero_prob:
        loss_zero_log_prob = criterion_neg_log_bernoulli(
            outputs["mlm_zero_probs"], target_values, masked_positions
        )
        loss += loss_zero_log_prob
        metrics_to_log.update({"train/nzlp": loss_zero_log_prob.item()})

    # Gene expression prediction loss (MSE) for cell embedding
    if config.GEPC:
        loss_gepc = masked_mse_loss(
            outputs["mvc_output"], target_values, masked_positions
        )
        loss += loss_gepc
        metrics_to_log.update({"train/mvc": loss_gepc.item()})

    # Non zero log prob loss (only when predicting cell embedding)
    if config.GEPC and config.explicit_zero_prob:
        loss_gepc_zero_log_prob = criterion_neg_log_bernoulli(
            outputs["mvc_zero_probs"], target_values, masked_positions
        )
        loss += loss_gepc_zero_log_prob
        metrics_to_log.update(
            {"train/mvc_nzlp": loss_gepc_zero_log_prob.item()}
        )

    if config.CLS:
        loss_cls = criterion_cls(outputs["cls_output"], celltype_labels)
        loss += loss_cls

        metrics_to_log.update({"train/cls": loss_cls.item()})

    if config.ECS:
        loss_ecs = 10 * outputs["loss_ecs"]
        loss += loss_ecs
        metrics_to_log.update({"train/ecs": loss_ecs.item()})

    if config.DAR:
        # try weighting and separate optimizer
        loss_dab = criterion_dab(outputs["dab_output"], batch_labels)
        loss += config.dab_weight * loss_dab
        metrics_to_log.update({"train/dab": loss_dab.item()})

    return loss, metrics_to_log


def train(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader,
    vocab: GeneVocab,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    accelerator: Accelerator,
):
    """Train the model for a single epoch"""

    # TODO: get mask value from GeneVocab
    mask_value = vocab["<mask>"]  # -1

    # Training loop
    for batch_idx, batch in enumerate(dataloader):
        input_gene_ids = batch["gene_ids"]
        input_values = batch["values"]
        target_values = batch["target_values"]
        batch_labels = batch["batch_labels"]
        celltype_labels = batch["celltype_labels"]
        
        # Forward pass
        outputs = model(
            input_gene_ids,
            input_values,
            src_key_padding_mask=input_gene_ids.eq(vocab[config.pad_token]),
            batch_labels=batch_labels,
            MVC=config.GEPC,
            ECS=config.ecs_thres > 0,
        )
        
        # Masked positions to predict
        masked_positions = input_values.eq(mask_value)

        loss, metrics_to_log = compute_loss(
            outputs=outputs,
            target_values=target_values, 
            masked_positions=masked_positions,
            celltype_labels=celltype_labels,
            batch_labels=batch_labels
        )

        # Backward pass and optimization
        accelerator.backward(loss)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


def main():
    args = parse_args()

    if args.vocab_file is None:
        vocab = get_default_gene_vocab()
    else:
        # TODO: GeneVocab should have a default built in
        vocab = GeneVocab.from_file(args.vocab_file)

    # TODO: Special tokens should already be in vocab
    special_tokens = [config.pad_token, "<cls>", "<eoc>"]

    # Load data
    train_dataloader, eval_dataloader = load_and_tokenize_data(args, vocab)

    # Initialize model
    model = init_model(vocab)

    # Initialize Adam optimizer and 
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, eps=1e-4 if config.amp else 1e-8
    )
    # Initialize learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        1, 
        gamma=config.schedule_ratio
    )

    # Initialize Accelerate Accelerator (multi-gpu training)
    # Track model metrics with wandb
    accelerator = Accelerator(log_with="wandb")
    model, optimizer, loader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    accelerator.init_trackers()

    for epoch in range(config.epochs):
        # Perform model training
        train(model=model, 
              vocab=vocab, 
              dataloader=train_dataloader,
              optimizer=optimizer,
              scheduler=scheduler,
              accelerator=accelerator)

        # Perform model evaluation
        evaluate(model=model, 
                 vocab=vocab, 
                 dataloader=eval_dataloader,
                  optimizer=optimizer,
                  scheduler=scheduler,
                 accelerator=accelerator)
