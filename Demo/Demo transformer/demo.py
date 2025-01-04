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

# 3. Extrahiere Embeddings
with torch.no_grad():
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state  # (1, seq_len, hidden_size)

# 4. Query, Key, Value durch lineare Projektionen erzeugen
W_Q = torch.nn.Linear(768, 64)  # Reduziere Dimensionen für bessere Übersicht
W_K = torch.nn.Linear(768, 64)
W_V = torch.nn.Linear(768, 64)

Q = W_Q(embeddings)  # (1, seq_len, 64)
K = W_K(embeddings)  # (1, seq_len, 64)
V = W_V(embeddings)  # (1, seq_len, 64)

# 5. Attention Scores berechnen
attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(64)
attention_weights = F.softmax(attention_scores, dim=-1)

# 6. Wende Attention auf V an
output = torch.matmul(attention_weights, V)

# Extrahiere Attention Matrix und Token
attention_matrix = attention_weights.squeeze().detach().numpy()
tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

# Identifiziere das Token "bank"
bank_token_idx = tokens.index("bank")

# 7. Visualisierung der Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(attention_matrix, annot=True, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
plt.title('Attention Matrix Heatmap')
plt.savefig('attention_plot_with_bank.png')
plt.close()

# 8. Visualisierung der Tokens mit Verbindungen
fig, ax = plt.subplots(figsize=(12, 12))

# Vertikale Positionen der Tokens links und rechts (umgedreht, damit der Satz von oben nach unten gelesen wird)
y_positions_left = np.arange(len(tokens) - 1, -1, -1)
y_positions_right = np.arange(len(tokens) - 1, -1, -1)

# Zeichne die Tokens links und rechts
ax.scatter(np.zeros(len(tokens)), y_positions_left, c='black', label='Tokens (Links)', s=100)
ax.scatter(np.ones(len(tokens)), y_positions_right, c='black', label='Tokens (Rechts)', s=100)

# Erstelle eine Farbmap, um Attention-Werte in Farben zu übersetzen
cmap = cm.viridis  # Wir verwenden die 'viridis' Farbskala für eine visuelle Ähnlichkeit mit der Heatmap

# Normalisiere die Attention-Gewichte für die Farbzuweisung
norm = plt.Normalize(vmin=np.min(attention_matrix), vmax=np.max(attention_matrix))

# Zeichne Verbindungen mit Farbkodierung
for i in range(len(tokens)):
    for j in range(len(tokens)):
        weight = attention_matrix[i, j]
        if weight > 0.05:  # Nur stärkere Verbindungen anzeigen
            color = cmap(norm(weight))  # `norm(weight)` gibt den Farbwert aus der Farbkarte zurück
            ax.plot([0, 1], [y_positions_left[i], y_positions_right[j]], color=color, lw=2)
            # Zeige den Attention Score auf der Linie an
            ax.text((0 + 1) / 2, (y_positions_left[i] + y_positions_right[j]) / 2,
                    f'{weight:.2f}', color='black', ha='center', va='center', fontsize=10)

# Setze Achsenlabels
ax.set_yticks(y_positions_left)
ax.set_yticklabels(tokens)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Links', 'Rechts'])
ax.set_xlabel('Tokens')

# Speichern der Visualisierung
plt.title('Token-Verbindungen mit Farbkodierung und Attention Scores')
plt.savefig('token_connections_with_attention_scores.png')
plt.close()

# 9. Neue Visualisierung der Verbindungen von "Bank"
fig, ax = plt.subplots(figsize=(12, 12))

# Zeichne die Tokens links und rechts
ax.scatter(np.zeros(len(tokens)), y_positions_left, c='black', label='Tokens (Links)', s=100)
ax.scatter(np.ones(len(tokens)), y_positions_right, c='black', label='Tokens (Rechts)', s=100)

# Zeichne nur die Verbindungen von "Bank"
for j in range(len(tokens)):
    weight = attention_matrix[bank_token_idx, j]
    if weight > 0.05:  # Nur stärkere Verbindungen anzeigen
        color = cmap(norm(weight))  # `norm(weight)` gibt den Farbwert aus der Farbkarte zurück
        ax.plot([0, 1], [y_positions_left[bank_token_idx], y_positions_right[j]], color=color, lw=2)
        # Zeige den Attention Score auf der Linie an
        ax.text((0 + 1) / 2, (y_positions_left[bank_token_idx] + y_positions_right[j]) / 2,
                f'{weight:.2f}', color='black', ha='center', va='center', fontsize=10)

# Setze Achsenlabels
ax.set_yticks(y_positions_left)
ax.set_yticklabels(tokens)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Links', 'Rechts'])
ax.set_xlabel('Tokens')

# Speichern der neuen Visualisierung
plt.title('Verbindungen von "Bank" zu anderen Tokens mit Attention Scores')
plt.savefig('connections_from_bank_only_with_scores.png')
plt.close()

print("Plots gespeichert als 'token_connections_with_attention_scores.png' und 'connections_from_bank_only_with_scores.png'")
