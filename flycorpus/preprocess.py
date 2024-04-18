import argparse
import gc
import os
from pathlib import Path
import scanpy as sc

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
        "--input-dir",
        "-i",
        type=str,
        required=True,
        help="Directory containing AnnData objects",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./data.scb",
        help="Directory to save scBank data, by default will make a "
        "directory named data.scb in the current directory",
    )
    # vocabulary
    parser.add_argument(
        "--vocab-file",
        "-v",
        type=str,
        default=None,
        help="File containing the gene vocabulary, default to None. "
        "If None, will use the default gene vocabulary from flyGPT, "
        "which use FlyBase gene symbols.",
    )
    return parser.parse_args()


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

    for file in files:
        adata = sc.read_loom(file, cache=True)
        print(f"Reading data from {file.name}: {adata.shape}")

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

    # Copy all parquet files to a single directory
    parquet_dir = outputdir / "datatable.parquet"
    parquet_dir.mkdir(exist_ok=True)
    print(f"Copying all parquet files to {parquet_dir}")
    for file in files:
        parquet = outputdir / f"{file.stem}.scb" / "X.datatable.parquet"
        if parquet.exists():
            os.symlink(parquet, parquet_dir / f"{file.stem}.datatable.parquet")
