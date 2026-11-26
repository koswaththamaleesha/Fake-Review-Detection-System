"""
data_loader.py
Loads and standardises both datasets into a unified DataFrame with columns:
    text   : str  - raw review text
    label  : int  - 1 = deceptive/fake, 0 = genuine
    source : str  - 'op_spam' | 'yelp'
    split  : str  - 'train' | 'test_indomain' | 'test_cross'
"""
from __future__ import annotations

import logging
import pathlib
import re

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Op-Spam (Deceptive Opinion Spam Corpus v1.4)
# Folder structure:
#   op_spam_v1.4/{positive,negative}_polarity/{deceptive_from_MTurk,truthful_from_*}/fold{1..5}/*.txt
# We use folds 1-4 for train/val, fold5 for in-domain test.
# ---------------------------------------------------------------------------
OPSPAM_ROOT = pathlib.Path(
    "data/op_spam_v1.4/op_spam_v1.4"
)
YELP_CSV = pathlib.Path("data/yelp_filtered/yelp_reviews.csv")

OPSPAM_TEST_FOLD = "fold5"


def load_opspam(root: pathlib.Path = OPSPAM_ROOT) -> pd.DataFrame:
    """Parse the op_spam folder tree into a tidy DataFrame."""
    records: list[dict] = []
    for polarity_dir in root.iterdir():
        if not polarity_dir.is_dir():
            continue
        polarity = polarity_dir.name  # positive_polarity / negative_polarity
        for label_dir in polarity_dir.iterdir():
            if not label_dir.is_dir():
                continue
            deceptive = "deceptive" in label_dir.name  # True = fake
            for fold_dir in label_dir.iterdir():
                if not fold_dir.is_dir():
                    continue
                fold = fold_dir.name  # fold1 … fold5
                for txt_file in fold_dir.glob("*.txt"):
                    # Extract hotel name from filename (e.g., d_hilton_1.txt)
                    hotel = re.sub(r"^[dt]_|_\d+\.txt$", "", txt_file.name)
                    text = txt_file.read_text(encoding="utf-8", errors="replace").strip()
                    records.append(
                        {
                            "text": text,
                            "label": 1 if deceptive else 0,
                            "source": "op_spam",
                            "polarity": polarity,
                            "hotel": hotel,
                            "fold": fold,
                            "split": (
                                "test_indomain" if fold == OPSPAM_TEST_FOLD else "train"
                            ),
                        }
                    )
    df = pd.DataFrame(records)
    logger.info("Op-Spam loaded: %d rows (train=%d, test=%d)", len(df),
                (df.split == "train").sum(), (df.split == "test_indomain").sum())
    return df


# ---------------------------------------------------------------------------
# Yelp Filtered Reviews (yelp_reviews.csv)
# flagged: 'Y' = spam/fake → label 1, 'N' = genuine → label 0
# Class imbalance: N≈20752, Y≈6206  → ratio ~3.3:1
# Strategy: use class_weight="balanced" in sklearn models (no resampling
#            at load time so the same DataFrame works for all strategies).
# Cross-dataset role: Yelp is held out as the cross-dataset test set.
# ---------------------------------------------------------------------------
def load_yelp(csv_path: pathlib.Path = YELP_CSV, sample_genuine: int | None = None) -> pd.DataFrame:
    """
    Load the Yelp filtered review CSV.

    Parameters
    ----------
    sample_genuine : int or None
        If set, randomly down-sample genuine reviews to this count so the
        overall dataset is more balanced for quick experiments.
        Leave None to keep all rows.
    """
    df_raw = pd.read_csv(csv_path)
    # Drop rows missing review text
    df_raw = df_raw.dropna(subset=["reviewContent"]).copy()
    df_raw["label"] = (df_raw["flagged"] == "Y").astype(int)
    df_raw["source"] = "yelp"
    df_raw["split"] = "test_cross"  # hold-out for cross-dataset eval

    if sample_genuine is not None:
        genuine = df_raw[df_raw.label == 0].sample(sample_genuine, random_state=42)
        fake = df_raw[df_raw.label == 1]
        df_raw = pd.concat([genuine, fake]).sample(frac=1, random_state=42)

    out = df_raw[["text" if "text" in df_raw.columns else "reviewContent",
                  "label", "source", "split"]].copy()
    out = out.rename(columns={"reviewContent": "text"})

    logger.info(
        "Yelp loaded: %d rows  (genuine=%d, fake=%d)",
        len(out),
        (out.label == 0).sum(),
        (out.label == 1).sum(),
    )
    return out


def load_all(
    opspam_root: pathlib.Path = OPSPAM_ROOT,
    yelp_csv: pathlib.Path = YELP_CSV,
) -> pd.DataFrame:
    """
    Returns a single combined DataFrame with all splits labelled:
        op_spam rows  → split ∈ {'train', 'test_indomain'}
        yelp rows     → split = 'test_cross'
    """
    op = load_opspam(opspam_root)[["text", "label", "source", "split"]]
    yelp = load_yelp(yelp_csv)
    combined = pd.concat([op, yelp], ignore_index=True)
    logger.info("Combined dataset: %d rows total", len(combined))
    logger.info("Split counts:\n%s", combined.groupby("split")["label"].value_counts().to_string())
    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_all()
    print(df.groupby(["split", "source", "label"]).size().reset_index(name="count").to_string(index=False))
