# -*- coding: utf-8 -*-
"""Homework_5_Q3 &4ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1o8py2nR27erZDK6XYUtMLRw39qSwfIso

Without Attention
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example dataset
english_french_data = [
("I am cold", "J'ai froid"),
("You are tired", "Tu es fatigué"),
("He is hungry", "Il a faim"),
("She is happy", "Elle est heureuse"),
("We are friends", "Nous sommes amis"),
("They are students", "Ils sont étudiants"),
("The cat is sleeping", "Le chat dort"),
("The sun is shining", "Le soleil brille"),
("We love music", "Nous aimons la musique"),
("She speaks French fluently", "Elle parle français couramment"),
("He enjoys reading books", "Il aime lire des livres"),
("They play soccer every weekend", "Ils jouent au football chaque week-end"),
("The movie starts at 7 PM", "Le film commence à 19 heures"),
("She wears a red dress", "Elle porte une robe rouge"),
("We cook dinner together", "Nous cuisinons le dîner ensemble"),
("He drives a blue car", "Il conduit une voiture bleue"),
("They visit museums often", "Ils visitent souvent des musées"),
("The restaurant serves delicious food", "Le restaurant sert une délicieuse cuisine"),
("She studies mathematics at university", "Elle étudie les mathématiques à l'université"),
("We watch movies on Fridays", "Nous regardons des films le vendredi"),
("He listens to music while jogging", "Il écoute de la musique en faisant du jogging"),
("They travel around the world", "Ils voyagent autour du monde"),
("The book is on the table", "Le livre est sur la table"),
("She dances gracefully", "Elle danse avec grâce"),
("We celebrate birthdays with cake", "Nous célébrons les anniversaires avec un gâteau"),
("He works hard every day", "Il travaille dur tous les jours"),
("They speak different languages", "Ils parlent différentes langues"),
("The flowers bloom in spring", "Les fleurs fleurissent au printemps"),
("She writes poetry in her free time", "Elle écrit de la poésie pendant son temps libre"),
("We learn something new every day", "Nous apprenons quelque chose de nouveau chaquejour"),
("The dog barks loudly", "Le chien aboie bruyamment"),
("He sings beautifully", "Il chante magnifiquement"),
("They swim in the pool", "Ils nagent dans la piscine"),
("The birds chirp in the morning", "Les oiseaux gazouillent le matin"),
("She teaches English at school", "Elle enseigne l'anglais à l'école"),
("We eat breakfast together", "Nous prenons le petit déjeuner ensemble"),
("He paints landscapes", "Il peint des paysages"),
("They laugh at the joke", "Ils rient de la blague"),
("The clock ticks loudly", "L'horloge tic-tac bruyamment"),
("She runs in the park", "Elle court dans le parc"),
("We travel by train", "Nous voyageons en train"),
("He writes a letter", "Il écrit une lettre"),
("They read books at the library", "Ils lisent des livres à la bibliothèque"),
("The baby cries", "Le bébé pleure"),
("She studies hard for exams", "Elle étudie dur pour les examens"),
("We plant flowers in the garden", "Nous plantons des fleurs dans le jardin"),
("He fixes the car", "Il répare la voiture"),
("They drink coffee in the morning", "Ils boivent du café le matin"),
("The sun sets in the evening", "Le soleil se couche le soir"),
("She dances at the party", "Elle danse à la fête"),
("We play music at the concert", "Nous jouons de la musique au concert"),
("He cooks dinner for his family", "Il cuisine le dîner pour sa famille"),
("They study French grammar", "Ils étudient la grammaire française"),
("The rain falls gently", "La pluie tombe doucement"),
("She sings a song", "Elle chante une chanson"),
("We watch a movie together", "Nous regardons un film ensemble"),
("He sleeps deeply", "Il dort profondément"),
("They travel to Paris", "Ils voyagent à Paris"),
("The children play in the park", "Les enfants jouent dans le parc"),
("She walks along the beach", "Elle se promène le long de la plage"),
("We talk on the phone", "Nous parlons au téléphone"),
("He waits for the bus", "Il attend le bus"),
("They visit the Eiffel Tower", "Ils visitent la tour Eiffel"),
("The stars twinkle at night", "Les étoiles scintillent la nuit"),
("She dreams of flying", "Elle rêve de voler"),
("We work in the office", "Nous travaillons au bureau"),
("He studies history", "Il étudie l'histoire"),
("They listen to the radio", "Ils écoutent la radio"),
("The wind blows gently", "Le vent souffle doucement"),
("She swims in the ocean", "Elle nage dans l'océan"),
("We dance at the wedding", "Nous dansons au mariage"),
("He climbs the mountain", "Il gravit la montagne"),
("They hike in the forest", "Ils font de la randonnée dans la forêt"),
("The cat meows loudly", "Le chat miaule bruyamment"),
("She paints a picture", "Elle peint un tableau"),
("We build a sandcastle", "Nous construisons un château de sable"),
("He sings in the choir", "Il chante dans le chœur"),
("They ride bicycles", "Ils font du vélo"),
("The coffee is hot", "Le café est chaud"),
("She wears glasses", "Elle porte des lunettes"),
("We visit our grandparents", "Nous rendons visite à nos grands-parents"),
("He plays the guitar", "Il joue de la guitare"),
("They go shopping", "Ils font du shopping"),
("The teacher explains the lesson", "Le professeur explique la leçon"),
("She takes the train to work", "Elle prend le train pour aller au travail"),
("We bake cookies", "Nous faisons des biscuits"),
("He washes his hands", "Il se lave les mains"),
("They enjoy the sunset", "Ils apprécient le coucher du soleil"),
("The river flows calmly", "La rivière coule calmement"),
("She feeds the cat", "Elle nourrit le chat"),
("We visit the museum", "Nous visitons le musée"),
("He fixes his bicycle", "Il répare son vélo"),
("They paint the walls", "Ils peignent les murs"),
("The baby sleeps peacefully", "Le bébé dort paisiblement"),
("She ties her shoelaces", "Elle attache ses lacets"),
("We climb the stairs", "Nous montons les escaliers"),
("He shaves in the morning", "Il se rase le matin"),
("They set the table", "Ils mettent la table"),
("The airplane takes off", "L'avion décolle"),
("She waters the plants", "Elle arrose les plantes"),
("We practice yoga", "Nous pratiquons le yoga"),
("He turns off the light", "Il éteint la lumière"),
("They play video games", "Ils jouent aux jeux vidéo"),
("The soup smells delicious", "La soupe sent délicieusement bon"),
("She locks the door", "Elle ferme la porte à clé"),
("We enjoy a picnic", "Nous profitons d'un pique-nique"),
("He checks his email", "Il vérifie ses emails"),
("They go to the gym", "Ils vont à la salle de sport"),
("The moon shines brightly", "La lune brille intensément"),
("She catches the bus", "Elle attrape le bus"),
("We greet our neighbors", "Nous saluons nos voisins"),
("He combs his hair", "Il se peigne les cheveux"),
("They wave goodbye", "Ils font un signe d'adieu")
]

# Special tokens
SOS_token = 0
EOS_token = 1
PAD_token = 2

# Building vocabularies
def build_vocab(sentences):
    word_to_idx = {"<sos>": SOS_token, "<eos>": EOS_token, "<pad>": PAD_token}
    idx_to_word = {SOS_token: "<sos>", EOS_token: "<eos>", PAD_token: "<pad>"}

    for sentence in sentences:  # Iterate over individual sentences
        for word in sentence.split():
            if word not in word_to_idx:
                idx = len(word_to_idx)
                word_to_idx[word] = idx
                idx_to_word[idx] = word

    return word_to_idx, idx_to_word

eng_vocab, eng_idx_to_word = build_vocab([pair[0] for pair in english_french_data])
fr_vocab, fr_idx_to_word = build_vocab([pair[1] for pair in english_french_data])

# Tokenization functions
def tokenize(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in sentence.split() if word in vocab] + [vocab["<eos>"]]

# Custom dataset
class TranslationDataset(Dataset):
    def __init__(self, data, eng_vocab, fr_vocab):
        self.data = data
        self.eng_vocab = eng_vocab
        self.fr_vocab = fr_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng_sentence, fr_sentence = self.data[idx]
        return torch.tensor(tokenize(eng_sentence, self.eng_vocab)), \
               torch.tensor(tokenize(fr_sentence, self.fr_vocab))

# Padding function for batch processing
def collate_fn(batch):
    eng_batch, fr_batch = zip(*batch)
    eng_batch = pad_sequence(eng_batch, padding_value=PAD_token, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_token, batch_first=True)
    return eng_batch.to(device), fr_batch.to(device)

# Split dataset into train/validation
dataset = TranslationDataset(english_french_data, eng_vocab, fr_vocab)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size=256, num_heads=2, num_layers=1, forward_expansion=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_dim, embed_size)
        self.decoder_embedding = nn.Embedding(output_dim, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src).permute(1, 0, 2)  # (seq_len, batch, embed_size)
        tgt = self.decoder_embedding(tgt).permute(1, 0, 2)
        out = self.transformer(src, tgt)
        return self.fc_out(out).permute(1, 0, 2)  # (batch, seq_len, output_dim)

# Training and validation functions
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for eng_batch, fr_batch in train_loader:
            optimizer.zero_grad()
            output = model(eng_batch, fr_batch[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), fr_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for eng_batch, fr_batch in val_loader:
                output = model(eng_batch, fr_batch[:, :-1])
                loss = criterion(output.reshape(-1, output.shape[-1]), fr_batch[:, 1:].reshape(-1))
                val_loss += loss.item()

                predicted = output.argmax(dim=-1)
                correct += (predicted == fr_batch[:, 1:]).sum().item()
                total += fr_batch[:, 1:].numel()

        val_accuracy = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_accuracy:.4f}")

# Experiment with different Transformer architectures
# Store the last trained model
best_model = None

layers = [1, 2, 4]
heads = [2, 4]
for num_layers in layers:
    for num_heads in heads:
        print(f"\nTraining Transformer with {num_layers} layers and {num_heads} heads")
        model = Transformer(input_dim=len(eng_vocab), output_dim=len(fr_vocab), num_layers=num_layers, num_heads=num_heads)
        train_model(model, train_loader, val_loader)
        best_model = model  # Save the last model (or apply selection logic)


# Inference function for translation
# Updated Inference function for translation
def translate(sentence):
    model.eval()  # Set model to evaluation mode
    tokens = tokenize(sentence, eng_vocab)  # Tokenize the input sentence
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)  # Add batch dimension and move to device
    tgt_tensor = torch.tensor([fr_vocab["<sos>"]]).unsqueeze(0).to(device)  # Initialize the target sequence with <sos> token

    translated_sentence = []  # Store translated words

    with torch.no_grad():  # Disable gradient calculation during inference
        for _ in range(10):  # Max translation length (can be adjusted)
            output = model(src_tensor, tgt_tensor)  # Get model predictions
            next_word = output.argmax(2)[:, -1].item()  # Get the predicted next word (highest probability)

            # Append the predicted word to the translated sentence
            translated_sentence.append(fr_idx_to_word[next_word])

            # If the model predicts the <eos> token, stop the translation
            if next_word == fr_vocab["<eos>"]:
                break

            # Update tgt_tensor by appending the predicted word for the next time step
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_word]]).to(device)], dim=1)

    # Clean the sentence by removing <sos>, <eos>, and <pad> tokens
    translated_sentence = [word for word in translated_sentence if word not in {fr_idx_to_word[SOS_token], fr_idx_to_word[EOS_token], fr_idx_to_word[PAD_token]}]

    # Join words into a single translated string and return it
    return " ".join(translated_sentence)

# Example translation
sentence_to_translate = "I am cold"
translated_output = translate(sentence_to_translate)
print("Translation:", translated_output)

"""Problem 4

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sample data (from your provided dataset)
english_french_data = [
    ("I am cold", "J'ai froid"),
    ("You are tired", "Tu es fatigué"),
    ("He is hungry", "Il a faim"),
    # ... (Use your full 100+ sentence dataset here)
]

# Special tokens
SOS_token, EOS_token, PAD_token = 0, 1, 2

# Vocabulary builder
def build_vocab(sentences):
    word_to_idx = {"<sos>": SOS_token, "<eos>": EOS_token, "<pad>": PAD_token}
    idx_to_word = {SOS_token: "<sos>", EOS_token: "<eos>", PAD_token: "<pad>"}
    for sentence in sentences:
        for word in sentence.split():
            if word not in word_to_idx:
                idx = len(word_to_idx)
                word_to_idx[word] = idx
                idx_to_word[idx] = word
    return word_to_idx, idx_to_word

eng_vocab, eng_idx_to_word = build_vocab([pair[0] for pair in english_french_data])
fr_vocab, fr_idx_to_word = build_vocab([pair[1] for pair in english_french_data])

# Tokenizer
def tokenize(sentence, vocab):
    return [vocab["<sos>"]] + [vocab[word] for word in sentence.split() if word in vocab] + [vocab["<eos>"]]

# Dataset class
class TranslationDataset(Dataset):
    def __init__(self, data, eng_vocab, fr_vocab):
        self.data = data
        self.eng_vocab = eng_vocab
        self.fr_vocab = fr_vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        eng, fr = self.data[idx]
        return torch.tensor(tokenize(eng, self.eng_vocab)), torch.tensor(tokenize(fr, self.fr_vocab))

# Collate function
def collate_fn(batch):
    eng_batch, fr_batch = zip(*batch)
    eng_batch = pad_sequence(eng_batch, padding_value=PAD_token, batch_first=True)
    fr_batch = pad_sequence(fr_batch, padding_value=PAD_token, batch_first=True)
    return eng_batch.to(device), fr_batch.to(device)

# Prepare DataLoaders
dataset = TranslationDataset(english_french_data, eng_vocab, fr_vocab)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collate_fn)

# Transformer Model
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, embed_size=256, num_heads=2, num_layers=1, forward_expansion=4, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(input_dim, embed_size)
        self.decoder_embedding = nn.Embedding(output_dim, embed_size)
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=embed_size * forward_expansion,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_size, output_dim)

    def forward(self, src, tgt):
        src = self.encoder_embedding(src).permute(1, 0, 2)
        tgt = self.decoder_embedding(tgt).permute(1, 0, 2)
        out = self.transformer(src, tgt)
        return self.fc_out(out).permute(1, 0, 2)

# Training function
def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_token)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for eng_batch, fr_batch in train_loader:
            optimizer.zero_grad()
            output = model(eng_batch, fr_batch[:, :-1])
            loss = criterion(output.reshape(-1, output.shape[-1]), fr_batch[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for eng_batch, fr_batch in val_loader:
                output = model(eng_batch, fr_batch[:, :-1])
                loss = criterion(output.reshape(-1, output.shape[-1]), fr_batch[:, 1:].reshape(-1))
                val_loss += loss.item()
                predicted = output.argmax(dim=-1)
                correct += (predicted == fr_batch[:, 1:]).sum().item()
                total += fr_batch[:, 1:].numel()

        val_acc = correct / total
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val Loss={val_loss/len(val_loader):.4f}, Val Acc={val_acc:.4f}")
    return model, val_acc

# Architecture exploration
best_model = None
best_accuracy = 0
results = {}

layers = [1, 2, 4]
heads = [2, 4]

for num_layers in layers:
    for num_heads in heads:
        print(f"\n➡️ Training Transformer with {num_layers} layers and {num_heads} heads...")
        model = Transformer(input_dim=len(eng_vocab), output_dim=len(fr_vocab), num_layers=num_layers, num_heads=num_heads)
        trained_model, val_acc = train_model(model, train_loader, val_loader)
        config_key = f"Layers:{num_layers}_Heads:{num_heads}"
        results[config_key] = val_acc
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            best_model = trained_model

# Show best result
print("\n✅ Best Transformer Config:")
for k, v in results.items():
    print(f"{k} -> Accuracy: {v:.4f}")
def translate(sentence, model):
    model.eval()
    tokens = tokenize(sentence, eng_vocab)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)
    tgt_tensor = torch.tensor([fr_vocab["<sos>"]]).unsqueeze(0).to(device)

    translated = []
    with torch.no_grad():
        for _ in range(20):
            output = model(src_tensor, tgt_tensor)
            next_token = output.argmax(2)[:, -1].item()
            if next_token == EOS_token:
                break
            translated.append(fr_idx_to_word.get(next_token, "<unk>"))
            tgt_tensor = torch.cat([tgt_tensor, torch.tensor([[next_token]]).to(device)], dim=1)

    return " ".join(translated)

# Test it
print("\n🗣️ Translating: 'He is hungry'")
print("French:", translate("He is hungry", best_model))