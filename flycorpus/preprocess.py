import argparse
import gc
import os
from pathlib import Path
import scanpy as sc
from datasets import Dataset, load_dataset

from flygpt.scbank import DataBank
from flygpt.tokenizer import GeneVocab, get_default_gene_vocab


"""command line example
python preprocess.py \
    --input-dir ./datasets/ \
    --output-dir ./databanks/ \
    --vocab-file ../../flygpt/tokenizer/default_flybase_vocab.json
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build large-scale data in scBank format "
        "from a group of AnnData objects in .loom format"
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing AnnData objects",
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


def loom_to_scbank(file: Path, vocab: GeneVocab, outputdir: Path) -> None:
    """
    Convert .loom file to .parquet using scBank
    """
    adata = sc.read_loom(file)
    print(f"Reading data from {file.name}: {adata.shape}")
    print(adata)

    db = DataBank.from_anndata(
            adata,
            vocab=vocab,
            to=outputdir / f"{file.stem}.scb",
            main_table_key="X",
            token_col="var_names",
            immediate_save=False,
        )

    db.meta_info.on_disk_format = "parquet"
    # Sync to disk
    db.sync()

    # Clean up
    del adata
    del db
    gc.collect()


def _prepend_cls(
        dataset: Dataset, 
        vocab: GeneVocab,
        pad_value: int = 0,
        num_proc: int = 1,
    ) -> Dataset:
    dataset = dataset.map(
        lambda example: {
            "genes": [vocab["<cls>"]] + example["genes"],
            "expressions": [pad_value] + example["expressions"],
        },
        # batched=True,  # not using since then the map func needs to loop
        num_proc=num_proc,
    )

    return dataset


def main():
    args = parse_args()
    inputdir = Path(args.input_dir)
    outputdir = Path(args.output_dir)

    files = [f for f in inputdir.glob("*.loom")]
    print(f"Found {len(files)} files in {inputdir}")

    if args.vocab_file is None:
        vocab = get_default_gene_vocab()
    else:
        # TODO: GeneVocab should have a default built in
        vocab = GeneVocab.from_file(args.vocab_file)

    print("pad", vocab["<pad>"])
    print("mask", vocab["<mask>"])
    print("cls", vocab["<cls>"])

    for file in files:
        print(f"Reading {file.name}")
        loom_to_scbank(file, vocab, outputdir)

    parquet_files = [
        str(parquet) for parquet in outputdir.glob("**/*.parquet")
    ]

    # Concatenate all parquet files
    dataset = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
    )

    # Prepend <cls> token to all cell representations
    dataset = _prepend_cls(
        dataset=dataset, 
        vocab=vocab, 
        pad_value=vocab["<pad>"],
        num_proc=10
    )

    # Save dataset to disk
    dataset.save_to_disk(outputdir / "cls_prefix_dataset.parquet")


if __name__ == "__main__":
    main()
