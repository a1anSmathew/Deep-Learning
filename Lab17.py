import pandas as pd
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib.pyplot as plt

path_sample_submission = "CAFA-6/sample_submission.tsv"
path_ia = "CAFA-6/IA.tsv"
path_test_taxon = "CAFA-6/Test/testsuperset-taxon-list.tsv"
path_train_terms = "CAFA-6/Train/train_terms.tsv"
path_train_taxonomy = "CAFA-6/Train/train_taxonomy.tsv"

# fasta files
path_test_fasta = "CAFA-6/Test/testsuperset.fasta"
path_train_fasta = "CAFA-6/Train/train_sequences.fasta"

# obo file
path_go_basic = "CAFA-6/Train/go-basic.obo"

# output
path_output_submission = "CAFA-6/submission.tsv"

train_terms = pd.read_csv(path_train_terms, sep="\t")
print("GO terms shape:", train_terms.shape)
print(train_terms.head())

#Load protein sequences from FASTA
records = list(SeqIO.parse(path_train_fasta, "fasta"))
seq_df = pd.DataFrame({
    "EntryID": [r.id.split("|")[1] if "|" in r.id else r.id for r in records],
    "sequence": [str(r.seq) for r in records]
})
print("Sequences shape:", seq_df.shape)
print(seq_df.head())

train_merged = train_terms.merge(seq_df, on="EntryID", how="left")
train_merged.dropna(subset=["sequence"], inplace=True)
print("✅ Merged dataset shape:", train_merged.shape)
train_merged.head()

model_name = "facebook/esm2_t6_8M_UR50D"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

def extract_embedding(seq):
    """Extracts mean-pooled embedding from an amino acid sequence."""
    inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

#demo: only sample a few hundred due to runtime
sample_df = train_merged.sample(300, random_state=42).reset_index(drop=True)

embeddings = []
for seq in tqdm(sample_df["sequence"], desc="Embedding Sequences"):
    try:
        emb = extract_embedding(seq)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error embedding: {e}")
        embeddings.append(np.zeros(320))

embeddings = np.vstack(embeddings)
print("Embedding matrix shape:", embeddings.shape)

# Group GO terms per protein
multi_label_df = (
    sample_df.groupby("EntryID")["term"]
    .apply(list)
    .reset_index()
    .merge(sample_df[["EntryID", "sequence"]].drop_duplicates(), on="EntryID")
)

mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(multi_label_df["term"])
print("Label matrix shape:", Y.shape)
print("Example labels:", multi_label_df["term"].iloc[0])

X_train, X_val, y_train, y_val = train_test_split(
    embeddings, Y, test_size=0.2, random_state=42
)

clf = OneVsRestClassifier(LogisticRegression(max_iter=200))
clf.fit(X_train, y_train)

y_pred = clf.predict_proba(X_val)
print("✅ Model trained.")

def fmax_score(y_true, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y_true.ravel(), y_pred.ravel())
    f_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    return np.nanmax(f_scores)

fmax = fmax_score(y_val, y_pred)
print(f"Fmax Score (Validation): {fmax:.4f}")

def read_submission_flex(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split()
            # Skip empty or invalid lines
            if len(parts) < 2:
                continue
            # Handle lines with 2 or 3 columns
            if len(parts) == 2:
                protein, go_term = parts
                score = np.nan
            elif len(parts) >= 3:
                protein, go_term, score = parts[:3]
            rows.append((protein, go_term, score))
    df = pd.DataFrame(rows, columns=["protein_id", "go_term", "score"])
    return df

# Load sample submission safely
submission = read_submission_flex(path_sample_submission)
print("Loaded sample_submission safely:", submission.shape)
print(submission.head())

# Generate dummy prediction scores for testing
submission["score"] = np.random.uniform(0.5, 0.9, len(submission))

# Save submission file
submission.to_csv(path_output_submission, sep="\t", index=False)
print("Submission file created successfully at:", path_output_submission)