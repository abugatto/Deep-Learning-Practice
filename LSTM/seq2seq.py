import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Vocabulary:
    def __init__(self, pad_token="<pad>", unk_token='<unk>'):
        self.id_to_string = {}
        self.string_to_id = {}
        
        # add the default pad token
        self.id_to_string[0] = pad_token
        self.string_to_id[pad_token] = 0
        
        # add the default unknown token
        self.id_to_string[1] = unk_token
        self.string_to_id[unk_token] = 1        
        
        # shortcut access
        self.pad_id = 0
        self.unk_id = 1
        
    def __len__(self):
        return len(self.id_to_string)

    def add_new_word(self, string):
        self.string_to_id[string] = len(self.string_to_id)
        self.id_to_string[len(self.id_to_string)] = string

    # Given a string, return ID
    def get_idx(self, string, extend_vocab=False):
        if string in self.string_to_id:
            return self.string_to_id[string]
        elif extend_vocab:  # add the new word
            self.add_new_word(string)
            return self.string_to_id[string]
        else:
            return self.unk_id


# Read the raw txt file and generate a 1D PyTorch tensor
# containing the whole text mapped to sequence of token IDs, and a vocab object.
class TextData:

    def __init__(self, file_path, vocab=None, extend_vocab=True, device='cuda'):
        self.data, self.vocab, self.fulltext = self.text_to_data(file_path, vocab, extend_vocab, device)
        
    def __len__(self):
        return len(self.data)

    def text_to_data(self, text_file, vocab, extend_vocab, device):
        """Read a raw text file and create its tensor and the vocab.

        Args:
          text_file: a path to a raw text file.
          vocab: a Vocab object
          extend_vocab: bool, if True extend the vocab
          device: device

        Returns:
          Tensor representing the input text, vocab file

        """
        assert os.path.exists(text_file)
        if vocab is None:
            vocab = Vocabulary()

        data_list = []

        # Construct data
        full_text = []
        print(f"Reading text file from: {text_file}")
        lines = 0
        with open(text_file, 'r') as text:
            for line in text:
                lines = lines + 1
                tokens = list(line)
                for token in tokens:
                    # get index will extend the vocab if the input
                    # token is not yet part of the text.
                    full_text.append(vocab.get_idx(token, extend_vocab=extend_vocab))
        print(f'\nNumber of Lines: {lines}\n')

        # convert to tensor
        data = torch.tensor(full_text, device=device, dtype=torch.int64)
        print("Done.")

        return data, vocab, full_text
    

# Since there is no need for schuffling the data, we just have to split
# the text data according to the batch size and bptt length.
# The input to be fed to the model will be batch[:-1]
# The target to be used for the loss will be batch[1:]
class DataBatches:
    def __init__(self, data, bsz, bptt_len, pad_id):
        self.batches = self.create_batch(data, bsz, bptt_len, pad_id)

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]

    def create_batch(self, input_data, bsz, bptt_len, pad_id):
        """Create batches from a TextData object .

        Args:
          input_data: a TextData object.
          bsz: int, batch size
          bptt_len: int, bptt length
          pad_id: int, ID of the padding token

        Returns:
          List of tensors representing batches

        """
        batches = []  # each element in `batches` is (len, B) tensor
        text_len = len(input_data)
        segment_len = text_len // bsz + 1 #computes the integer division of text_len / batch size + 1
        print(segment_len)
        print(segment_len * bsz)

        # Question: Explain the next two lines!
        padded = input_data.data.new_full((segment_len * bsz,), pad_id) #vector of padded tokens
        print(f'Length of padded and data is: [{padded.shape}, {input_data.data.shape}]')
        padded[:text_len] = input_data.data #copy 
        padded = padded.view(bsz, segment_len).t()
        num_batches = segment_len // bptt_len + 1

        for i in range(num_batches):
            # Prepare batches such that the last symbol of the current batch
            # is the first symbol of the next batch.
            if i == 0:
                # Append a dummy start symbol using pad token
                batch = torch.cat(
                    [padded.new_full((1, bsz), pad_id),
                     padded[i * bptt_len:(i + 1) * bptt_len]], dim=0)
                batches.append(batch)
            else:
                batches.append(padded[i * bptt_len - 1:(i + 1) * bptt_len])

        return batches

# downlaod the text
# Make sure to go to the link and check how the text looks like.
#!wget http://www.gutenberg.org/files/49010/49010-0.txt

DEVICE = 'cuda'

batch_size = 32
bptt_len = 64 #backpropogation through time

my_data = TextData(text_path, device=DEVICE)
batches = DataBatches(my_data, batch_size, bptt_len, pad_id=0)

#Answer 1.1:
vocab = my_data.vocab
text = my_data.fulltext
data = my_data.data
print(f'Number of Tokens: {vocab.__len__()}')
print(vocab.id_to_string)
print(vocab.string_to_id)
print(f'\nNumber of Tokens in Dataset: {data.__len__()}')
print(data[0:100])
print(f'\nNumber of Batches: {batches.__len__()}')
print(f'Batch Shape [bptt,batch_sz]: {batches[0].shape}')
print(f'Total Batches (64 lists): {batches[0]}')
print(f'Batch 2 (32 tokens): {batches[0][0:2]}')

# input to the network
print(batches[0][:-1].shape)
print(batches[0][:-1])

# target tokens to be predicted
print(batches[0][1:].shape)
print(batches[0][1:])

# RNN based language model (predicts next token from current token and memory)
class RNNModel(nn.Module):
    def __init__(self, num_classes, emb_dim, hidden_dim, num_layers, test=False):
        """Parameters:
        
          num_classes (int): number of input/output classes
          emb_dim (int): token embedding size
          hidden_dim (int): hidden layer size of RNNs
          num_layers (int): number of RNN layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.test = test

        #embedding layer translates index tokens into 64 byte embeddings
        # the embedding weights are learned like a fully connected layer
        # it is different from fully connected because it expects the index instead of one hot encodings
        self.input_layer = nn.Embedding(num_classes, emb_dim) #[vocab_size,embedding]
        self.rnn = nn.RNN(emb_dim, hidden_dim, num_layers) #[embedding,hidden_dim]
        self.out_layer = nn.Linear(hidden_dim, num_classes) #[hidden_dim,vocab_size]

    def forward(self, input, state):
        if self.test: print(f'RNN input: {input.shape}')
        emb = self.input_layer(input)
        if self.test: print(f'RNN embedding input: {emb.shape}')
        output, state = self.rnn(emb, state)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes) #returns tensor with [batch_size,num_classes]
        if self.test: print(f'RNN output: {emb.shape}')
        return output, state

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, bsz, self.hidden_dim)


# To be modified for LSTM...
def custom_detach(h):
    return h.detach()

@torch.no_grad()
def complete(model, prompt, steps, sample=False, test = False):
    """Complete the prompt for as long as given steps using the model.
    
    Parameters:
      model: language model
      prompt (str): text segment to be completed
      steps (int): number of decoding steps.
      sample (bool): If True, sample from the model. Otherwise greedy.

    Returns:
      completed text (str)
    """
    model.eval()
    out_list = []
    
    # forward the prompt, compute prompt's ppl
    prompt_list = []
    char_prompt = list(prompt)
    for char in char_prompt:
        prompt_list.append(my_data.vocab.string_to_id[char])
    x = torch.tensor(prompt_list).to(DEVICE).unsqueeze(1)

    if test: print(f'Prompt: {x.shape}')
    if test: print(prompt)
    if test: print(x.t())
    
    states = model.init_hidden(1)
    logits, states = model(x, states)
    probs = F.softmax(logits[-1], dim=-1)

    if test: print(f'States Shape: {states.shape}')
    if test: print(f'Probs Shape (one for each char in Vocabulary): {probs.shape}')
    if test: print(probs)
        
    if sample:
        out = torch.multinomial(probs, num_samples=1).item()
    else:
        out = torch.argmax(probs).item()

    out_list.append(my_data.vocab.id_to_string[int(out)])
    x = torch.ones((1,1)).new_full((1,1), out, dtype=torch.int64).to(DEVICE)
    
    # decode 
    for k in range(steps):
        logits, states = model(x, states)
        probs = F.softmax(logits, dim=-1)
        if sample:  # sample from the distribution or take the most likely
            out = torch.multinomial(probs, num_samples=1).item()
        else:
            out = torch.argmax(probs).item()
        out_list.append(my_data.vocab.id_to_string[int(out)])
        x = torch.ones((1,1)).new_full((1,1),out,dtype=torch.int64).to(DEVICE)
    return ''.join(out_list)

learning_rate = 0.0005
clipping = 1.0
embedding_size = 64
rnn_size = 2048
rnn_num_layers = 1

# vocab_size = len(module.vocab.itos)
vocab_size = len(my_data.vocab.id_to_string)
print(F"vocab size: {vocab_size}")

model = RNNModel(num_classes=vocab_size, emb_dim=embedding_size, hidden_dim=rnn_size, num_layers=rnn_num_layers, test=False)
model = model.to(DEVICE)
hidden = model.init_hidden(batch_size)
print(model)

loss_fn = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

# Training
test= False
num_epochs = 30
report_every = 30
prompt = "Dogs like best to"

perplexities = []
best_perplexity = np.inf
for ep in range(num_epochs):
    print(f"=== start epoch {ep} ===")
    state = model.init_hidden(batch_size)
    perplexity = np.inf

    #Model must be training each batch sequentially. Not sure how parallelism would work
    for idx in range(len(batches)):
        batch = batches[idx]
        model.train()
        optimizer.zero_grad()

        #Detach from computational graph so new model won't backprop further than 64 segments
        state = custom_detach(state)
        
        #Batches input this way so that each sequence in each batch would run in parallel
        # this allows the data to not need shuffling because they are already inplicitly shuffled
        # WHY ISN'T IT BETTER TO GO THROUGH CONTIGUOUS SEQUENCES SO MEMORY CAN BE LEARNED???
        input = batch[:-1] #input batches 0 to N-1
        target = batch[1:].reshape(-1) #target batches are 1 to N

        bsz = input.shape[1]
        prev_bsz = state.shape[1] 
        if test: print(f'Old and new batch sizes: [{prev_bsz}, {bsz}]')
        if bsz != prev_bsz:
            state = state[:, :bsz, :]
        output, state = model(input, state)
        loss = loss_fn(output, target)
        perplexity = np.exp(loss.item())
        if perplexity < best_perplexity: best_perplexity = perplexity

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping)
        optimizer.step()
        if idx % report_every == 0 or test:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train perplexity: {perplexity}")
            generated_text = complete(model, prompt, 128, sample=False, test=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt)
            print(generated_text)
            print(f'----------------- end generated text -------------------')\

        if test: break

    perplexities.append(perplexity)
    if test: break

complete(model, "THE DONKEY IN THE LION’S SKIN", 512, sample=False)

#Greedy Decoding for various prompts:
#A title of a fable which exists in the book.
complete(model, "THE DONKEY AND THE FROGS", 512, sample=False)

#A title which you invent, which is not in the book, but similar in the style.
complete(model, "THE MONKEY AND THE MAN'S SON", 512, sample=False)

#Some texts in a similar style.
complete(model, "THE SANDS AND THEIR TIME", 512, sample=False)

#Anything you think might be interesting.
complete(model, "complete(model, \"THE DONKEY IN THE LION’S SKIN\", 512)", 512, sample=False)

#Display perplexities
import matplotlib.pyplot as plt

plt.plot(range(len(perplexities)), perplexities, color='blue', label='Perplexity')
plt.axhline(best_perplexity, color='red', linestyle='--', label=f'Best Perplexity: {best_perplexity : .2f}')
plt.legend()
plt.title(f'Perplexity of the RNN Language Model')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.savefig('perp.png')

# LSTM based language model (predicts next token from current token and memory)
class LSTMModel(nn.Module):
    def __init__(self, num_classes, emb_dim, hidden_dim, num_layers, test=False):
        """Parameters:
        
          num_classes (int): number of input/output classes
          emb_dim (int): token embedding size
          hidden_dim (int): hidden layer size of RNNs
          num_layers (int): number of RNN layers
        """
        super().__init__()
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        #embedding layer translates index tokens into 64 byte embeddings
        # the embedding weights are learned like a fully connected layer
        # it is different from fully connected because it expects the index instead of one hot encodings
        self.input_layer = nn.Embedding(num_classes, emb_dim) #[vocab_size,embedding]
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers) #[embedding,hidden_dim]
        self.out_layer = nn.Linear(hidden_dim, num_classes) #[hidden_dim,vocab_size]

    def forward(self, input, prev):
        emb = self.input_layer(input)
        output, (state, cell) = self.lstm(emb, prev)
        output = self.out_layer(output)
        output = output.view(-1, self.num_classes) #returns tensor with [batch_size,num_classes]
        return output, (state, cell)

    def init_hidden(self, bsz):
        state = torch.zeros(self.num_layers, bsz, self.hidden_dim).to(DEVICE)

        cell = torch.zeros(self.num_layers, bsz, self.hidden_dim).to(DEVICE)

        return state, cell

#Detaches state and cell
def custom_detach(state, cell):
    return state.detach(), cell.detach()

batch_size2 = 32
bptt2 = 64
learning_rate2 = 0.001
clipping2 = 1.0
embedding_size2 = 64
lstm_size = 2048
lstm_num_layers = 1

# vocab_size = len(module.vocab.itos)
vocab_size = len(my_data.vocab.id_to_string)
print(F"vocab size: {vocab_size}")

model2 = LSTMModel(num_classes=vocab_size, emb_dim=embedding_size2, hidden_dim=lstm_size, num_layers=lstm_num_layers, test=False)
model2 = model2.to(DEVICE)
hidden2 = model2.init_hidden(batch_size2)
print(model2)

loss_fn2 = nn.CrossEntropyLoss(ignore_index=0)
optimizer2 = torch.optim.Adam(params=model2.parameters(), lr=learning_rate2)

# Training
test2 = False
num_epochs2 = 30
report_every2 = 30
prompt2 = "Dogs like best to"

perplexities2 = []
best_perplexity2 = np.inf
for ep in range(num_epochs2):
    print(f"=== start epoch {ep} ===")
    state, cell = model2.init_hidden(batch_size)
    perplexity2 = np.inf

    #Model must be training each batch sequentially. Not sure how parallelism would work
    for idx in range(len(batches)):
        batch = batches[idx]
        model2.train()
        optimizer2.zero_grad()

        #Detach from computational graph so new model won't backprop further than 64 segments
        # and will not compute gradients
        state, cell = custom_detach(state, cell)
        
        #Batches input this way so that each sequence in each batch would run in parallel
        # this allows the data to not need shuffling because they are already inplicitly shuffled
        input = batch[:-1] #input batches 0 to N-1
        target = batch[1:].reshape(-1) #target batches are 1 to N

        bsz = input.shape[1]
        prev_bsz = state.shape[1] 
        if test2: print(f'Old and new batch sizes: [{prev_bsz}, {bsz}]')
        if bsz != prev_bsz:
            state = state[:, :bsz, :]
        output, (state, cell) = model2(input, (state, cell))
        loss = loss_fn2(output, target)
        perplexity2 = np.exp(loss.item())
        if perplexity2 < best_perplexity2: best_perplexity2 = perplexity2

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), clipping2)
        optimizer2.step()
        if idx % report_every2 == 0 or test2:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train perplexity: {perplexity2}")
            generated_text = complete(model2, prompt2, 128, sample=False, test=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt2)
            print(generated_text)
            print(f'----------------- end generated text -------------------')\

        if test2: break

    perplexities2.append(perplexity2)
    if test2: break

complete(model2, "THE DONKEY IN THE LION’S SKIN", 512, sample=False)

#Greedy Decoding for: A title of a fable which exists in the book.
complete(model2, "THE DONKEY AND THE FROGS", 512, sample=False)

#Greedy Decoding for: A title which you invent, which is not in the book, but similar in the style.
complete(model2, "THE SANDS AND THEIR TIME", 512, sample=False)

#Sampling for: A title which you invent, which is not in the book, but similar in the style.
complete(model2, "THE SANDS AND THEIR TIME", 512, sample=True)

#Display perplexities
import matplotlib.pyplot as plt

plt.plot(range(len(perplexities2)), perplexities2, color='blue', label='Perplexity')
plt.axhline(best_perplexity2, color='red', linestyle='--', label=f'Best Perplexity: {best_perplexity2 : .2f}')
plt.legend()
plt.title(f'Perplexity of the RNN Language Model')
plt.xlabel('Epochs')
plt.ylabel('Perplexity')
plt.savefig('LSTMperp.png')

#Get new dataset from python 
batch_size3 = 32
bptt3 = 64

text_path3 = "/content/harry.txt"
my_data3 = TextData(text_path3, device=DEVICE)
batches3 = DataBatches(my_data3, batch_size, bptt_len, pad_id=0)

learning_rate3 = 0.001
clipping3 = 1.0
embedding_size3 = 64
lstm_size3 = 2048
lstm_num_layers3 = 1

# vocab_size = len(module.vocab.itos)
vocab_size3 = len(my_data3.vocab.id_to_string)
print(my_data3.vocab.id_to_string)
print(f"vocab size: {vocab_size3}")

model3 = LSTMModel(num_classes=vocab_size3, emb_dim=embedding_size3, hidden_dim=lstm_size3, num_layers=lstm_num_layers3, test=False)
model3 = model3.to(DEVICE)
hidden3 = model3.init_hidden(batch_size3)
print(model3)

loss_fn3 = nn.CrossEntropyLoss(ignore_index=0)
optimizer3 = torch.optim.Adam(params=model3.parameters(), lr=learning_rate3)

# Training
test3 = False
num_epochs3 = 30
report_every3 = 30
prompt3 = "self.conv4 = nn.Conv2d(64, 64, 3)"

perplexities3 = []
best_perplexity3 = np.inf
for ep in range(num_epochs3):
    print(f"=== start epoch {ep} ===")
    state, cell = model3.init_hidden(batch_size3)
    perplexity3 = np.inf

    #Model must be training each batch sequentially. Not sure how parallelism would work
    for idx in range(len(batches3)):
        batch = batches3[idx]
        model3.train()
        optimizer3.zero_grad()

        #Detach from computational graph so new model won't backprop further than 64 segments
        # and will not compute gradients
        state, cell = custom_detach(state, cell)
        
        #Batches input this way so that each sequence in each batch would run in parallel
        # this allows the data to not need shuffling because they are already inplicitly shuffled
        input = batch[:-1] #input batches 0 to N-1
        target = batch[1:].reshape(-1) #target batches are 1 to N

        bsz = input.shape[1]
        prev_bsz = state.shape[1] 
        if test3: print(f'Old and new batch sizes: [{prev_bsz}, {bsz}]')
        if bsz != prev_bsz:
            state = state[:, :bsz, :]
        output, (state, cell) = model3(input, (state, cell))
        loss = loss_fn3(output, target)
        perplexity3 = np.exp(loss.item())
        if perplexity3 < best_perplexity3: best_perplexity3 = perplexity3

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model3.parameters(), clipping3)
        optimizer3.step()
        if idx % report_every3 == 0 or test3:
            #print(f"train loss: {loss.item()}")  # replace me by the line below!
            print(f"train perplexity: {perplexity3}")
            generated_text = complete(model3, prompt3, 128, sample=False, test=False)
            print(f'----------------- epoch/batch {ep}/{idx} -----------------')
            print(prompt3)
            print(generated_text)
            print(f'----------------- end generated text -------------------')\

        if test3: break

    perplexities3.append(perplexity3)
    if test2: break

#Greedy Decoding for
complete(model2, "Dudly said", 512, sample=False)

#Greedy Decoding for 
complete(model2, "Harry said", 512, sample=False)

#Sampling for: A title which you invent, which is not in the book, but similar in the style.
complete(model2, "snake", 512, sample=True)