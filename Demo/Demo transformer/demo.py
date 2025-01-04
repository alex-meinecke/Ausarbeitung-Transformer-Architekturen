import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np

# 1. Initialisierung von Tokenizer und Modell
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 2. Eingabesatz
sentence = "Ich sitze auf der Bank"
inputs = tokenizer(sentence, return_tensors='pt')

# 3. Extrahiere Embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_size)

print("Embeddings Shape:", embeddings.shape)  # (1, seq_len, 768)

# 4. Query, Key, Value durch lineare Projektionen erzeugen
W_Q = torch.nn.Linear(768, 64)  # Reduziere Dimensionen für bessere Übersicht
W_K = torch.nn.Linear(768, 64)
W_V = torch.nn.Linear(768, 64)

Q = W_Q(embeddings)  # (1, seq_len, 64)
K = W_K(embeddings)  # (1, seq_len, 64)
V = W_V(embeddings)  # (1, seq_len, 64)

print("Q Shape:", Q)
print("K Shape:", K)
print("V Shape:", V)

# 5. Attention Scores berechnen
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(64)
attention_weights = F.softmax(attention_scores, dim=-1)

# 6. Wende Attention auf V an
output = torch.matmul(attention_weights, V)

print("Attention Weights Shape:", attention_weights.shape)  # (1, seq_len, seq_len)
print("Attention Output Shape:", output.shape)  # (1, seq_len, 64)

import matplotlib.pyplot as plt
import seaborn as sns

attention_matrix = attention_weights.squeeze().detach().numpy()
sns.heatmap(attention_matrix, annot=True, cmap='viridis', xticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]), yticklabels=tokenizer.convert_ids_to_tokens(inputs['input_ids'][0]))
plt.savefig('attention_plot.png')
print("Plot gespeichert als attention_plot.png")
