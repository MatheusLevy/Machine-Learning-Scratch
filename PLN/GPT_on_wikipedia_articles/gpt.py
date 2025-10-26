import torch.nn as nn
import random
import torch.optim as optim
from tqdm import tqdm
import torch

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads  # embed_size: 256 and heads: 8 then each head is
                                             # 256 // 8 = 32 dim
        assert (self.head_dim * heads == embed_size), 'Embed size needs to be div by heads'
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias= False)
        self.keys =  nn.Linear(self.head_dim, self.head_dim, bias= False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias= False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)
        
    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        
        # Split embedding into self.head pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)
        
        energy = torch.einsum('nqhd,nkhd->nhqk', [queries, keys])
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, keys_len, heads, head_dim)
        # energy = (N, heads, query_len, key_len)
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e20'))
        
        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        
        out = torch.einsum('nhql,nlhd->nqhd', [attention, values]).reshape(
            N, query_len, self.heads*self.head_dim
        )
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, values_len, heads, heads_dim)
        # (N, query_len, heads, head_dim) then flatten last two dimension
        # (n, query_len, embed_size)
        out = self.fc_out(out)
        return out
    
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion*embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion*embed_size, embed_size)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, value, key, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, trg_mask)
        return out
    
class Decoder(nn.Module):
    def __init__(self,
                 trg_vocab_size,
                 embed_size,
                 num_layers,
                 heads,
                 forward_expansion,
                 dropout,
                 device,
                 max_lenght
                ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embedding = nn.Embedding(max_lenght, embed_size)
        self.layers = nn.ModuleList(
            [DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, trg_mask):
        N, seq_lenght = x.shape
        positions = torch.arange(0, seq_lenght).expand(N, seq_lenght).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.positional_embedding(positions)))
        for layer in self.layers:
            x = layer(x, x, x, trg_mask)
        
        out = self.fc_out(x)
        return out
    
class GPT(nn.Module):
    def __init__(
        self,
        trg_vocab_size,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads= 8,
        dropout=0,
        device='cuda',
        max_lenght=100
    ):
        super(GPT, self).__init__()
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_lenght   
        )
        self.trg_pad_ix = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)
    
    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N,1,trg_len, trg_len)
        return trg_mask.to(self.device)
    
    def forward(self, trg):
        trg_mask = self.make_trg_mask(trg)
        out = self.decoder(trg, trg_mask)
        return out
    
trg_vocab_size = 10
trg_pad_idx = 0
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT(trg_vocab_size, trg_pad_idx, device=device).to(device)

TRG_PAD_IDX = 0
criterion = nn.CrossEntropyLoss(ignore_index = TRG_PAD_IDX)
trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)
output = model(trg)
print(output.shape)
# output shape (N, seq_len, vocab_Size)
output_dim = output.shape[-1] #
output = output[:, 1:].contiguous().view(-1, output_dim) # shape (N * seq_len)
# trg shape = (N, seq_len)
print(trg.shape)
trg = trg[:, 1:].contiguous().view(-1) # Remove start token from stences so shape 
                                       # (N*seq_len - N)
loss = criterion(output, trg)
print(loss.item())
predicted_tokens = torch.argmax(output, dim=-1)
print(predicted_tokens)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
import random
from torchtext.vocab import build_vocab_from_iterator
pt = spacy.load("pt_core_news_sm")

def split_dataset(input_file, train_file, test_file, train_percent):
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()  

    total_lines = len(lines)
    train_lines = int(total_lines * train_percent)
    test_lines = total_lines - train_lines


    random.shuffle(lines)

    with open(train_file, 'w', encoding='utf-8') as train:
        for line in lines[:train_lines]:
            train.write(line)

    with open(test_file, 'w', encoding='utf-8') as test:
        for line in lines[train_lines:]:
            test.write(line)


input_file = r'C:\Users\levyb\Documents\Machine-Learning-Scratch\Pytorch\PLN\sentences_output_clean.txt'


train_file = r'C:\Users\levyb\Documents\Machine-Learning-Scratch\Pytorch\PLN\GPT\shiny_train.txt'
test_file = r'C:\Users\levyb\Documents\Machine-Learning-Scratch\Pytorch\PLN\GPT\shiny_test.txt'

train_percent = 0.8

split_dataset(input_file, train_file, test_file, train_percent)

FILE_PATH = train_file
data_pipe = dp.iter.IterableWrapper([FILE_PATH])
data_pipe = dp.iter.FileOpener(data_pipe, mode='rb')
data_pipe = data_pipe.parse_csv(skip_lines=0, delimiter='\n',)

for sample in data_pipe:
    print(sample)
    break

def ptTokenizer(text):
    return [tok.text for tok in pt.tokenizer(text)]

print(ptTokenizer("Ali ibne Abi Talibe (em árabe: علي بن أبي طالب; Meca c."))

def getTokens(data_iter):
    """
    Function to yield tokens from an iterator. Since, our iterator contains
    tuple of sentences (source and target), `place` parameters defines for which
    index to return the tokens for. `place=0` for source and `place=1` for target
    """
    for sentence in data_iter:
        yield ptTokenizer(sentence[0])

source_vocab = build_vocab_from_iterator(
    getTokens(data_pipe),
    min_freq=2,
    specials= ['<pad>', '<sos>', '<eos>', '<unk>'],
    special_first=True
)
source_vocab.set_default_index(source_vocab['<unk>'])

print(source_vocab.get_itos()[:9])

def getTransform(vocab):
    """
    Create transforms based on given vocabulary. The returned transform is applied to sequence
    of tokens.
    """
    text_tranform = T.Sequential(
        ## converts the sentences to indices based on given vocabulary
        T.VocabTransform(vocab=vocab),
        ## Add <sos> at beginning of each sentence. 1 because the index for <sos> in vocabulary is
        # 1 as seen in previous section
        T.AddToken(1, begin=True),
        ## Add <eos> at beginning of each sentence. 2 because the index for <eos> in vocabulary is
        # 2 as seen in previous section
        T.AddToken(2, begin=False)
    )
    return text_tranform

temp_list = list(data_pipe)
some_sentence = temp_list[798][0]
print("Some sentence= ", end="")
print(some_sentence)
transformed_sentence = getTransform(source_vocab)(ptTokenizer(some_sentence))
print("Transformed sentence=", end="")
print(transformed_sentence)
index_to_string = source_vocab.get_itos()
for index in transformed_sentence:
    print(index_to_string[index], end=" ")

def applyTransform(sentence):
    """
    Apply transforms to sequence of tokens in a sequence pair
    """

    return getTransform(source_vocab)(ptTokenizer(sentence[0]))

data_pipe = data_pipe.map(applyTransform) ## Apply the function to each element in the iterator
temp_list = list(data_pipe)

data_pipe = data_pipe.bucketbatch(
    batch_size = 32, batch_num=5,  bucket_num=1,
    use_in_batch_shuffle=False
)

def applyPadding(sentence):
    """
    Convert sequences to tensors and apply padding
    """
    return T.ToTensor(0)(list(sentence))

data_pipe = data_pipe.map(applyPadding)

trg_vocab_size = len(source_vocab)
trg_pad_idx = source_vocab.lookup_indices(['<pad>'])[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT(trg_vocab_size, trg_pad_idx, device=device).to(device)
criterion = nn.CrossEntropyLoss(ignore_index = trg_pad_idx)
optimizer = optim.Adam(model.parameters())
clip = 1

def calc_len_dp(dp):
    i=0 
    for _ in dp:
        i+=1
    return i

len_dp_train = calc_len_dp(data_pipe)
print(f"len_dp_train={len_dp_train}", end="")
scaler = torch.cuda.amp.GradScaler()

def train_gpt(model, iterator, optimizer, criterion, clip, epochs):

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        print(f'Epoch {epoch}/{epochs}')
        for idx, trg in enumerate(iterator):
            N, seq_len = trg.shape
            if seq_len > 100:
                continue

            # trg shape N, seq_len
            trg = trg[:, 1:]
            trg = trg.to(device)
            output = model(trg)

            output_dim = output.shape[-1]
            output = output[:].contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)

            if (idx==0) and (epoch!=0):

                predicted_tokens = torch.argmax(output, dim=-1)
                print('source')
                src_tokens = source_vocab.lookup_tokens(trg.tolist())
                print(' '.join(src_tokens[:src_tokens.index('<eos>')]))
                print('pred')
                pred_tokens = source_vocab.lookup_tokens(predicted_tokens.tolist())
                print(' '.join(pred_tokens[:pred_tokens.index('<eos>')]))

            optimizer.zero_grad()
            loss = criterion(output, trg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()

            epoch_loss += loss.item()
        print('Loss: ', epoch_loss/len_dp_train)

epochs = 5
train_gpt(model, data_pipe, optimizer, criterion, clip, epochs)

from torchmetrics.functional.text import bleu_score
import numpy as np
def bleu(data_pipe, model, device):
    targets = []
    outputs = []
    
    model.eval()
    for idx, trg in enumerate(data_pipe):

            
        with torch.no_grad():
            trg = torch.tensor(trg, device=device).unsqueeze(0)
             # trg shape N, seq_len
            trg = trg[:, 1:]
            N, seq_len = trg.shape
            trg = trg.to(device)
            output = model(trg)
            output_dim = output.shape[-1]
            output = output[:].contiguous().view(-1, output_dim)
            trg = trg.contiguous().view(-1)
            predicted_tokens = torch.argmax(output, dim=-1)
            src_tokens = source_vocab.lookup_tokens(trg.tolist())
            pred_tokens = source_vocab.lookup_tokens(predicted_tokens.tolist())
            
            targets.append(' '.join(src_tokens[:src_tokens.index('<eos>')]))
            outputs.append(' '.join(pred_tokens[:pred_tokens.index('<eos>')]))
            
    return bleu_score(outputs, targets)

FILE_PATH = r'C:\Users\levyb\Documents\Machine-Learning-Scratch\Pytorch\PLN\GPT\shiny_test.txt'
test_data_pipe = dp.iter.IterableWrapper([FILE_PATH])
test_data_pipe = dp.iter.FileOpener(test_data_pipe, mode='rb')
test_data_pipe = test_data_pipe.parse_csv()

test_data_pipe = test_data_pipe.map(applyTransform) ## Apply the function to each element in the iterator

score = bleu(test_data_pipe, model, device)
print(score)