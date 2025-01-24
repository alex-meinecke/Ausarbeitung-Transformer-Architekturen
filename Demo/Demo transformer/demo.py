import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# 1. Initialisierung von Tokenizer und Modell
tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
model = BertModel.from_pretrained('bert-base-german-cased')

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
    embeddings = outputs.last_hidden_state

# 4. Embeddingmatrix anzeigen (Dimensionen verkleinert)
embedding_matrix = embeddings.squeeze(0).numpy()
embedding_reduced = embedding_matrix[:, :5]  # Reduziere die Dimensionen
print("Embedding-Matrix mit reduzierten Dimensionen:")
print(embedding_reduced)

# Visualisierung der Embeddingmatrix
plt.figure(figsize=(8, 6))
sns.heatmap(embedding_reduced, annot=True, cmap='viridis', xticklabels=[f'Dim {i+1}' for i in range(embedding_reduced.shape[1])])
plt.title('Token Embedding Matrix (reduzierte Dimensionen)')
plt.xlabel('Embedding Dimensionen')
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
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(768)
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

# 8. Token-Verbindungen visualisieren für jeden Token
for i in range(len(tokens)):
    fig, ax = plt.subplots(figsize=(12, 12))
    y_positions = np.arange(len(tokens) - 1, -1, -1)

    # Zeichnen der Punkte und Verbindungen für den aktuellen Token
    ax.scatter(np.zeros(len(tokens)), y_positions, c='black', s=100)
    ax.scatter(np.ones(len(tokens)), y_positions, c='black', s=100)

    cmap = cm.viridis
    norm = plt.Normalize(vmin=np.min(attention_matrix), vmax=np.max(attention_matrix))

    # Zeichnen der Verbindungen für den aktuellen Token
    for j in range(len(tokens)):
        weight = attention_matrix[i, j]
        if weight > 0.1:  # Threshold für darzustellende Verbindungen
            color = cmap(norm(weight))
            ax.plot([0, 1], [y_positions[i], y_positions[j]], color=color, lw=2)
            ax.text(0.5, (y_positions[i] + y_positions[j]) / 2, f'{weight:.2f}', ha='center', va='center', fontsize=10)

    # Token-Namen nur an der Y-Achse links anbringen
    for idx, token in enumerate(tokens):
        ax.text(-0.1, y_positions[idx], token, ha='center', va='center', fontsize=12, color='black')

    # Token-Namen an der Y-Achse auf der rechten Seite anbringen
    for idx, token in enumerate(tokens):
        ax.text(1.1, y_positions[idx], token, ha='center', va='center', fontsize=12, color='black')

    ax.set_yticks(y_positions)
    ax.set_yticklabels([])  # Entferne die Y-Achsenbeschriftungen, da die Tokens jetzt manuell hinzugefügt werden
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Links', 'Rechts'])
    plt.title(f'Token-Verbindungen mit Attention Scores für Token {tokens[i]}')
    plt.savefig(f'token_{tokens[i]}_connections_with_attention_scores.png')
    plt.close()
    print(f"Token-Verbindungen für {tokens[i]} gespeichert als 'token_{tokens[i]}_connections_with_attention_scores.png'")

# 9. Alle Token-Verbindungen auf einmal visualisieren
fig, ax = plt.subplots(figsize=(12, 12))
y_positions = np.arange(len(tokens) - 1, -1, -1)

# Zeichnen der Punkte für die Tokens
ax.scatter(np.zeros(len(tokens)), y_positions, c='black', s=100)
ax.scatter(np.ones(len(tokens)), y_positions, c='black', s=100)

cmap = cm.viridis
norm = plt.Normalize(vmin=np.min(attention_matrix), vmax=np.max(attention_matrix))

# Zeichnen der Verbindungen zwischen allen Token-Paaren
for i in range(len(tokens)):
    for j in range(len(tokens)):
        weight = attention_matrix[i, j]
        if weight > 0.1:  # Threshold für darzustellende Verbindungen
            color = cmap(norm(weight))
            ax.plot([0, 1], [y_positions[i], y_positions[j]], color=color, lw=2)

# Token-Namen nur an der Y-Achse links anbringen
for idx, token in enumerate(tokens):
    ax.text(-0.1, y_positions[idx], token, ha='center', va='center', fontsize=12, color='black')

# Token-Namen an der Y-Achse auf der rechten Seite anbringen
for idx, token in enumerate(tokens):
    ax.text(1.1, y_positions[idx], token, ha='center', va='center', fontsize=12, color='black')

# Achsenticks und -labels
ax.set_yticks(y_positions)
ax.set_yticklabels([])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Links', 'Rechts'])
plt.title('Alle Token-Verbindungen')
plt.savefig('all_token_connections.png')
plt.close()
print("Alle Token-Verbindungen mit Attention Scores gespeichert als 'all_token_connections_with_attention_scores.png'")
