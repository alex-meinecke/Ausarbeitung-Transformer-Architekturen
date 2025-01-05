import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# 1. Initialisierung von Tokenizer und Modell
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 2. Eingabesatz
sentence = "Ich sitze auf der Bank"
inputs = tokenizer(sentence, return_tensors='pt')

# Token-IDs und Tokens ausgeben
print("Token-IDs:", inputs['input_ids'][0].tolist())
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
print("Tokens:", tokens)

# 3. Extrahiere Embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_size)

# 4. Embeddingmatrix anzeigen (Dimensionen verkleinert)
embedding_matrix = embeddings.squeeze(0).numpy()
embedding_reduced = embedding_matrix[:, :5]  # Reduziere die Dimensionen (z.B. nur die ersten 5 Dimensionen)
print("Embedding-Matrix mit reduzierten Dimensionen:")
print(embedding_reduced)

# Visualisierung der Embeddingmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(embedding_reduced, annot=True, cmap='viridis')
plt.title('Token Embedding Matrix (reduzierte Dimensionen)')
plt.xlabel('Embedding Dimensionen (Ausschnitt)')
plt.ylabel('Tokens')
plt.savefig('token_embedding_matrix_reduced.png')
plt.close()
print("Embeddingmatrix gespeichert als 'token_embedding_matrix_reduced.png'")

# 5. Query, Key, Value durch lineare Projektionen erzeugen
W_Q = torch.nn.Linear(768, 768)
W_K = torch.nn.Linear(768, 768)
W_V = torch.nn.Linear(768, 768)

Q = W_Q(embeddings)
K = W_K(embeddings)
V = W_V(embeddings)

# 6. Attention Scores berechnen
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(64)
attention_weights = F.softmax(attention_scores, dim=-1)

# Extrahiere Attention Matrix
attention_matrix = attention_weights.squeeze().detach().numpy()

# 7. Heatmap der Attention-Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, annot=True, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
plt.title('Attention Matrix Heatmap')
plt.savefig('attention_matrix_heatmap.png')
plt.close()
print("Attention-Heatmap gespeichert als 'attention_matrix_heatmap.png'")

# 8. Token-Verbindungen visualisieren
fig, ax = plt.subplots(figsize=(12, 12))
y_positions = np.arange(len(tokens) - 1, -1, -1)
ax.scatter(np.zeros(len(tokens)), y_positions, c='black', s=100)
ax.scatter(np.ones(len(tokens)), y_positions, c='black', s=100)

cmap = cm.viridis
norm = plt.Normalize(vmin=np.min(attention_matrix), vmax=np.max(attention_matrix))

for i in range(len(tokens)):
    for j in range(len(tokens)):
        weight = attention_matrix[i, j]
        if weight > 0.05:
            color = cmap(norm(weight))
            ax.plot([0, 1], [y_positions[i], y_positions[j]], color=color, lw=2)
            ax.text(0.5, (y_positions[i] + y_positions[j]) / 2, f'{weight:.2f}', ha='center', va='center', fontsize=10)

ax.set_yticks(y_positions)
ax.set_yticklabels(tokens)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Links', 'Rechts'])
plt.title('Token-Verbindungen mit Attention Scores')
plt.savefig('token_connections_with_attention_scores.png')
plt.close()
print("Token-Verbindungen gespeichert als 'token_connections_with_attention_scores.png'")

print("Alle Visualisierungen wurden erfolgreich gespeichert.")
