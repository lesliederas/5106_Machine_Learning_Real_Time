# -*- coding: utf-8 -*-
"""Homework4_Q3.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/14rLfkYXGeXzY43a9j2OlqBeo_rfpjJPa
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""French-> English

without Attention
"""

english_to_french = [
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
# Special tokens for the start and end of sequences
SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token

# Swap the pairs: French → English
french_to_english = [(fr, en) for en, fr in english_to_french]

# Update vocab mapping
word_to_index = {"SOS": SOS_token, "EOS": EOS_token}
for pair in french_to_english:
    for word in pair[0].split() + pair[1].split():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

index_to_seq = {i: word for word, i in word_to_index.items()}

#No changes needed! The same architecture can be used since it's language-agnostic.
class Translation(Dataset):
    """Custom Dataset class for handling translation pairs."""
    def __init__(self, dataset, word_to_index):
        self.dataset = dataset
        self.word_to_index = word_to_index

    def __len__(self):
        # Returns the total number of translation pairs in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves a translation pair by index, converts words to indices,
        # and adds the EOS token at the end of each sentence.
        input_sentence, target_sentence = self.dataset[idx]
        input_indices = [self.word_to_index[word] for word in input_sentence.split()] + [EOS_token]
        target_indices = [self.word_to_index[word] for word in target_sentence.split()] + [EOS_token]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

# Dataset and Dataloader
translation_dataset = Translation(french_to_english, word_to_index)
dataloader = DataLoader(translation_dataset, batch_size=1, shuffle=True)

#No changes needed! The same architecture can be used since it's language-agnostic.
class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU layer

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)

#No changes needed! The same architecture can be used since it's language-agnostic.

class Decoder(nn.Module):
    """The Decoder part of the seq2seq model."""
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU layer
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

input_size = len(word_to_index)
hidden_size = 256
output_size = len(word_to_index)

encoder = Encoder(input_size=input_size, hidden_size=hidden_size).to(device)
decoder = Decoder(hidden_size=hidden_size, output_size=output_size).to(device)


# Set the learning rate for optimization
learning_rate = 0.009


# Initializing optimizers for both encoder and decoder with Adam optimizer
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    # Initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    # Clear gradients for optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Calculate the length of input and target tensors
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    # Initialize loss
    loss = 0

    # Encoding each word in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Decoder starts with the encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoding loop
    for di in range(target_length):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        # Choose top1 word from decoder's output
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        # Calculate loss
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:  # Stop if EOS token is generated
            break

    # Backpropagation
    loss.backward()

    # Update encoder and decoder parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return average loss
    return loss.item() / target_length

# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 30

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)

        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss

    if epoch % 2 == 0:
        print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')


def evaluate_and_print():
    encoder.eval()
    decoder.eval()
    total_loss = 0
    correct = 0
    total_words = 0

    with torch.no_grad():
        for input_tensor, target_tensor in dataloader:
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()
            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            # Encode
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

            # Decode
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                total_loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if topi.item() == target_tensor[di].item():
                    correct += 1
                total_words += 1

                if decoder_input.item() == EOS_token:
                    break

    print(f"Validation Loss: {total_loss.item() / total_words:.4f}")
    print(f"Validation Accuracy: {correct / total_words * 100:.2f}%")

evaluate_and_print()
def translate_french_sentence(sentence):
    encoder.eval()
    decoder.eval()

    input_indices = [word_to_index[word] for word in sentence.split()] + [EOS_token]
    input_tensor = torch.tensor(input_indices, dtype=torch.long).to(device)

    encoder_hidden = encoder.initHidden()
    for ei in range(len(input_tensor)):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)

    decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_hidden = encoder_hidden

    output_words = []
    with torch.no_grad():
        for _ in range(20):  # max translation length
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
            topv, topi = decoder_output.topk(1)
            word_index = topi.item()
            if word_index == EOS_token:
                break
            output_words.append(index_to_seq[word_index])
            decoder_input = topi.squeeze().detach()

    return ' '.join(output_words)

# Test a few
sample_sentences = [
    "Elle porte une robe rouge",
    "Ils voyagent autour du monde",
    "Le chat dort",
    "Nous célébrons les anniversaires avec un gâteau"
]

for sentence in sample_sentences:
    print(f"French: {sentence}")
    print(f"Predicted English: {translate_french_sentence(sentence)}\n")

"""With Attention"""

french_to_english = [
    ("J'ai froid", "I am cold"),
    ("Tu es fatigué", "You are tired"),
    ("Il a faim", "He is hungry"),
    ("Elle est heureuse", "She is happy"),
    ("Nous sommes amis", "We are friends"),
    ("Ils sont étudiants", "They are students"),
    ("Le chat dort", "The cat is sleeping"),
    ("Le soleil brille", "The sun is shining"),
    ("Nous aimons la musique", "We love music"),
    ("Elle parle français couramment", "She speaks French fluently"),
    ("Il aime lire des livres", "He enjoys reading books"),
    ("Ils jouent au football chaque week-end", "They play soccer every weekend"),
    ("Le film commence à 19 heures", "The movie starts at 7 PM"),
    ("Elle porte une robe rouge", "She wears a red dress"),
    ("Nous cuisinons le dîner ensemble", "We cook dinner together"),
    ("Il conduit une voiture bleue", "He drives a blue car"),
    ("Ils visitent souvent des musées", "They visit museums often"),
    ("Le restaurant sert une délicieuse cuisine", "The restaurant serves delicious food"),
    ("Elle étudie les mathématiques à l'université", "She studies mathematics at university"),
    ("Nous regardons des films le vendredi", "We watch movies on Fridays"),
    ("Il écoute de la musique en faisant du jogging", "He listens to music while jogging"),
    ("Ils voyagent autour du monde", "They travel around the world"),
    ("Le livre est sur la table", "The book is on the table"),
    ("Elle danse avec grâce", "She dances gracefully"),
    ("Nous célébrons les anniversaires avec un gâteau", "We celebrate birthdays with cake"),
    ("Il travaille dur tous les jours", "He works hard every day"),
    ("Ils parlent différentes langues", "They speak different languages"),
    ("Les fleurs fleurissent au printemps", "The flowers bloom in spring"),
    ("Elle écrit de la poésie pendant son temps libre", "She writes poetry in her free time"),
    ("Nous apprenons quelque chose de nouveau chaque jour", "We learn something new every day"),
    ("Le chien aboie bruyamment", "The dog barks loudly"),
    ("Il chante magnifiquement", "He sings beautifully"),
    ("Ils nagent dans la piscine", "They swim in the pool"),
    ("Les oiseaux gazouillent le matin", "The birds chirp in the morning"),
    ("Elle enseigne l'anglais à l'école", "She teaches English at school"),
    ("Nous prenons le petit déjeuner ensemble", "We eat breakfast together"),
    ("Il peint des paysages", "He paints landscapes"),
    ("Ils rient de la blague", "They laugh at the joke"),
    ("L'horloge tic-tac bruyamment", "The clock ticks loudly"),
    ("Elle court dans le parc", "She runs in the park"),
    ("Nous voyageons en train", "We travel by train"),
    ("Il écrit une lettre", "He writes a letter"),
    ("Ils lisent des livres à la bibliothèque", "They read books at the library"),
    ("Le bébé pleure", "The baby cries"),
    ("Elle étudie dur pour les examens", "She studies hard for exams"),
    ("Nous plantons des fleurs dans le jardin", "We plant flowers in the garden"),
    ("Il répare la voiture", "He fixes the car"),
    ("Ils boivent du café le matin", "They drink coffee in the morning"),
    ("Le soleil se couche le soir", "The sun sets in the evening"),
    ("Elle danse à la fête", "She dances at the party"),
    ("Nous jouons de la musique au concert", "We play music at the concert"),
    ("Il cuisine le dîner pour sa famille", "He cooks dinner for his family"),
    ("Ils étudient la grammaire française", "They study French grammar"),
    ("La pluie tombe doucement", "The rain falls gently"),
    ("Elle chante une chanson", "She sings a song"),
    ("Nous regardons un film ensemble", "We watch a movie together"),
    ("Il dort profondément", "He sleeps deeply"),
    ("Ils voyagent à Paris", "They travel to Paris"),
    ("Les enfants jouent dans le parc", "The children play in the park"),
    ("Elle se promène le long de la plage", "She walks along the beach"),
    ("Nous parlons au téléphone", "We talk on the phone"),
    ("Il attend le bus", "He waits for the bus"),
    ("Ils visitent la tour Eiffel", "They visit the Eiffel Tower"),
    ("Les étoiles scintillent la nuit", "The stars twinkle at night"),
    ("Elle rêve de voler", "She dreams of flying"),
    ("Nous travaillons au bureau", "We work in the office"),
    ("Il étudie l'histoire", "He studies history"),
    ("Ils écoutent la radio", "They listen to the radio"),
    ("Le vent souffle doucement", "The wind blows gently"),
    ("Elle nage dans l'océan", "She swims in the ocean"),
    ("Nous dansons au mariage", "We dance at the wedding"),
    ("Il gravit la montagne", "He climbs the mountain"),
    ("Ils font de la randonnée dans la forêt", "They hike in the forest"),
    ("Le chat miaule bruyamment", "The cat meows loudly"),
    ("Elle peint un tableau", "She paints a picture"),
    ("Nous construisons un château de sable", "We build a sandcastle"),
    ("Il chante dans le chœur", "He sings in the choir"),
    ("Ils font du vélo", "They ride bicycles"),
    ("Le café est chaud", "The coffee is hot"),
    ("Elle porte des lunettes", "She wears glasses"),
    ("Nous rendons visite à nos grands-parents", "We visit our grandparents"),
    ("Il joue de la guitare", "He plays the guitar"),
    ("Ils font du shopping", "They go shopping"),
    ("Le professeur explique la leçon", "The teacher explains the lesson"),
    ("Elle prend le train pour aller au travail", "She takes the train to work"),
    ("Nous faisons des biscuits", "We bake cookies"),
    ("Il se lave les mains", "He washes his hands"),
    ("Ils apprécient le coucher du soleil", "They enjoy the sunset"),
    ("La rivière coule calmement", "The river flows calmly"),
    ("Elle nourrit le chat", "She feeds the cat"),
    ("Nous visitons le musée", "We visit the museum"),
    ("Il répare son vélo", "He fixes his bicycle"),
    ("Ils peignent les murs", "They paint the walls"),
    ("Le bébé dort paisiblement", "The baby sleeps peacefully"),
    ("Elle attache ses lacets", "She ties her shoelaces"),
    ("Nous montons les escaliers", "We climb the stairs"),
    ("Il se rase le matin", "He shaves in the morning"),
    ("Ils mettent la table", "They set the table"),
    ("L'avion décolle", "The airplane takes off"),
    ("Elle arrose les plantes", "She waters the plants"),
    ("Nous pratiquons le yoga", "We practice yoga"),
    ("Il éteint la lumière", "He turns off the light"),
    ("Ils jouent aux jeux vidéo", "They play video games"),
    ("La soupe sent délicieusement bon", "The soup smells delicious"),
    ("Elle ferme la porte à clé", "She locks the door"),
    ("Nous profitons d'un pique-nique", "We enjoy a picnic"),
    ("Il vérifie ses emails", "He checks his email"),
    ("Ils vont à la salle de sport", "They go to the gym"),
    ("La lune brille intensément", "The moon shines brightly"),
    ("Elle attrape le bus", "She catches the bus"),
    ("Nous saluons nos voisins", "We greet our neighbors"),
    ("Il se peigne les cheveux", "He combs his hair"),
    ("Ils font un signe d'adieu", "They wave goodbye")
]

# Special tokens for the start and end of sequences
SOS_token = 0  # Start Of Sequence Token
EOS_token = 1  # End Of Sequence Token
max_length = 12

# Preparing the word to index mapping and vice versa
word_to_index = {"SOS": SOS_token, "EOS": EOS_token}
for pair in french_to_english:
    for word in pair[0].split() + pair[1].split():
        if word not in word_to_index:
            word_to_index[word] = len(word_to_index)

index_to_seq = {i: word for word, i in word_to_index.items()}

class Translation(Dataset):
    """Custom Dataset class for handling translation pairs."""
    def __init__(self, dataset, word_to_index):
        self.dataset = dataset
        self.word_to_index = word_to_index

    def __len__(self):
        # Returns the total number of translation pairs in the dataset
        return len(self.dataset)

    def __getitem__(self, idx):
        # Retrieves a translation pair by index, converts words to indices,
        # and adds the EOS token at the end of each sentence.
        input_sentence, target_sentence = self.dataset[idx]
        input_indices = [self.word_to_index[word] for word in input_sentence.split()] + [EOS_token]
        target_indices = [self.word_to_index[word] for word in target_sentence.split()] + [EOS_token]
        return torch.tensor(input_indices, dtype=torch.long), torch.tensor(target_indices, dtype=torch.long)

# Creating a DataLoader to batch and shuffle the dataset
translation_dataset = Translation(french_to_english, word_to_index)
dataloader = DataLoader(translation_dataset, batch_size=1, shuffle=True)

class Encoder(nn.Module):
    """The Encoder part of the seq2seq model."""
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)  # Embedding layer
        self.gru = nn.GRU(hidden_size, hidden_size)  # GRU layer

    def forward(self, input, hidden):
        # Forward pass for the encoder
        embedded = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        # Initializes hidden state
        return torch.zeros(1, 1, self.hidden_size, device=device)

class AttnDecoder(nn.Module):
    """Decoder with attention mechanism using GRU."""
    def __init__(self, hidden_size, output_size, max_length=12, dropout_p=0.1):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        # Attention weights: concatenated [embedded + hidden] -> attention over max_length
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)

        # Combine embedded input and attention context
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)

        # GRU instead of LSTM
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)

        # Final output layer
        self.out = nn.Linear(self.hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Embedding layer
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        # Calculate attention weights
        attn_input = torch.cat((embedded[0], hidden[0]), 1)  # Shape: (1, hidden*2)
        attn_weights = F.softmax(self.attn(attn_input), dim=1)  # Shape: (1, max_length)

        # Apply attention weights to encoder outputs
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))  # Shape: (1, 1, hidden_size)

        # Combine embedded input with context vector
        output = torch.cat((embedded[0], attn_applied[0]), 1)  # Shape: (1, hidden*2)
        output = self.attn_combine(output).unsqueeze(0)       # Shape: (1, 1, hidden_size)
        output = F.relu(output)

        # GRU step
        output, hidden = self.gru(output, hidden)  # output: (1, 1, hidden_size), hidden: (1, 1, hidden_size)

        # Final output
        output = F.log_softmax(self.out(output[0]), dim=1)  # Shape: (1, output_size)

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
# Assuming all words in the dataset + 'SOS' and 'EOS' tokens are included in word_to_index
input_size = len(word_to_index)
hidden_size = 256  # Adjust according to your preference
output_size = len(word_to_index)

encoder = Encoder(input_size=input_size, hidden_size=hidden_size).to(device)
decoder = AttnDecoder(hidden_size=hidden_size, output_size=output_size).to(device)

# Set the learning rate for optimization
learning_rate = 0.009

# Initializing optimizers for both encoder and decoder with Adam optimizer
encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,max_length=12):
    # Initialize encoder hidden state
    encoder_hidden = encoder.initHidden()

    # Clear gradients for optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Calculate the length of input and target tensors
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    # Initialize loss
    loss = 0

    # Encoding each word in the input
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    # Decoder's first input is the SOS token
    decoder_input = torch.tensor([[SOS_token]], device=device)

    # Decoder starts with the encoder's last hidden state
    decoder_hidden = encoder_hidden

    # Decoding loop
    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        # Choose top1 word from decoder's output
        topv, topi = decoder_output.topk(1)
        decoder_input = topi.squeeze().detach()  # Detach from history as input

        # Calculate loss
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        if decoder_input.item() == EOS_token:  # Stop if EOS token is generated
            break

    # Backpropagation
    loss.backward()

    # Update encoder and decoder parameters
    encoder_optimizer.step()
    decoder_optimizer.step()

    # Return average loss
    return loss.item() / target_length

# Negative Log Likelihood Loss function for calculating loss
criterion = nn.NLLLoss()

# Set number of epochs for training
n_epochs = 30

# Training loop
for epoch in range(n_epochs):
    total_loss = 0
    for input_tensor, target_tensor in dataloader:
        # Move tensors to the correct device
        input_tensor = input_tensor[0].to(device)
        target_tensor = target_tensor[0].to(device)

        # Perform a single training step and update total loss
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        total_loss += loss

    # Print loss every 10 epochs
    if epoch % 2 == 0:
       print(f'Epoch {epoch}, Loss: {total_loss / len(dataloader)}')

def evaluate(encoder, decoder, dataloader, criterion, n_examples=10):
    # Switch model to evaluation mode
    encoder.eval()
    decoder.eval()

    total_loss = 0
    correct_predictions = 0

    # No gradient calculation
    with torch.no_grad():
        for i, (input_tensor, target_tensor) in enumerate(dataloader):
            # Move tensors to the correct device
            input_tensor = input_tensor[0].to(device)
            target_tensor = target_tensor[0].to(device)

            encoder_hidden = encoder.initHidden()

            input_length = input_tensor.size(0)
            target_length = target_tensor.size(0)

            loss = 0

            # Encoding step
            encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
            for ei in range(input_length):
                encoder_output, encoder_hidden = encoder(input_tensor[ei].unsqueeze(0), encoder_hidden)
                encoder_outputs[ei] = encoder_output[0, 0]

            # Decoding step
            decoder_input = torch.tensor([[SOS_token]], device=device)
            decoder_hidden = encoder_hidden

            predicted_indices = []

            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                predicted_indices.append(topi.item())
                decoder_input = topi.squeeze().detach()

                loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
                if decoder_input.item() == EOS_token:
                    break

            # Calculate and print loss and accuracy for the evaluation
            total_loss += loss.item() / target_length
            if predicted_indices == target_tensor.tolist():
                correct_predictions += 1

            # Optionally, print some examples
            if i < n_examples:
                predicted_sentence = ' '.join([index_to_seq[index] for index in predicted_indices if index not in (SOS_token, EOS_token)])
                target_sentence = ' '.join([index_to_seq[index.item()] for index in target_tensor if index.item() not in (SOS_token, EOS_token)])
                input_sentence = ' '.join([index_to_seq[index.item()] for index in input_tensor if index.item() not in (SOS_token, EOS_token)])

                print(f'Input: {input_sentence}, Target: {target_sentence}, Predicted: {predicted_sentence}')

        # Print overall evaluation results
        average_loss = total_loss / len(dataloader)
        accuracy = correct_predictions / len(dataloader)
        print(f'Evaluation Loss: {average_loss}, Accuracy: {accuracy}')

# Perform evaluation with examples
evaluate(encoder, decoder, dataloader, criterion)