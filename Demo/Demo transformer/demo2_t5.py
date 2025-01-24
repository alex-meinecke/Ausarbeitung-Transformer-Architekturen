from transformers import T5Tokenizer, T5ForConditionalGeneration

# Initialisiere Tokenizer und Modell
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# Eingabetext
input_text = "translate English to German: The weather is nice today."

# Tokenisiere den Eingabetext
input_ids = tokenizer(input_text, return_tensors="pt").input_ids

# Encoder-Verarbeitung: Eingabe durch den Encoder schicken
encoder_outputs = model.encoder(input_ids)

# Encoder-Ausgaben extrahieren
encoder_hidden_states = encoder_outputs.last_hidden_state

# Decoder-Verarbeitung: Ein Start-Token an den Decoder geben
decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids

# Decoder-Ausgaben generieren
decoder_outputs = model.decoder(
    input_ids=decoder_input_ids,
    encoder_hidden_states=encoder_hidden_states
)

# Decoder-Zwischenzustand extrahieren
decoder_hidden_states = decoder_outputs.last_hidden_state

# Komplette Sequenzgenerierung: Encoder und Decoder kombinieren
output_ids = model.generate(input_ids, max_length=40)

# Ausgabe-Text dekodieren
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

# Ausgabe formatieren
print("\n--- Ausgabe des Modells ---")
print(f"Eingabetext: {input_text}")
print(f"Übersetzung: {output_text}\n")

print("Encoder-Zwischenschicht:")
print(encoder_hidden_states.shape)  # Ausgabe der Form der Encoder-Zwischenschicht

print("\nDecoder-Zwischenschicht:")
print(decoder_hidden_states.shape)  # Ausgabe der Form der Decoder-Zwischenschicht

print("\nGenerierte IDs:")
print(output_ids)  # Ausgabe der generierten Token-IDs

print("\nDecoder Eingabe-IDs:")
print(decoder_input_ids)  # Ausgabe der Decoder Eingabe-IDs


