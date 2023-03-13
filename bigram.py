import torch
import torch.nn as nn
from torch.nn import functional as F

BATCH_SIZE = 32
BLOCK_SIZE = 8
MAX_ITERS = 3000
EVAL_INTERAL = 300
LEARNING_RATE = 1e-2
EVAL_ITERS = 200
N_EMBED = 32

torch.manual_seed(1337)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# Read training dataset
with open('input.txt', 'r') as f:
    text = f.read()

# Define the vocabulary
chars = sorted(list(set(text)))
VOCAB_SIZE = len(chars)

# Define dicts and methods to encode and decode text to/from the vocabulary
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Split data into train and test splits
data = torch.tensor(encode(text), dtype=torch.long, device=device)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# Extract random batches of text from data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    inputs = torch.stack([data[i:i+BLOCK_SIZE] for i in ix]).to(device)
    targets = torch.stack([data[i+1:i+BLOCK_SIZE+1] for i in ix]).to(device)
    return inputs, targets

# Define the Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()

        # Construct the embedding table which represents each token in the
        # vocabulary as a vector of learnable parameters. This table
        # is trained (modified) during the training process.

        # The addition of the linear layer lets us shrink dramatically the
        # size of the embedding table. This is a common technique to reduce
        # the number of parameters in a model. The embedding table is
        # typically very large (vocab_size x vocab_size) and this is
        # problematic for training. 
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, VOCAB_SIZE)
        # self.token_embedding_table = nn.Embedding(VOCAB_SIZE, N_EMBED)

        # This table is used to look up the token position in the block
        # self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBED)

        # New version adds a linear layer to map the embedding table to
        # the vocabulary size
        # self.lm_head = nn.Linear(N_EMBED, VOCAB_SIZE)

    def forward(self, idx, targets=None):

        # From previous version
        logits = self.token_embedding_table(idx) 

        # Given the input token, lookup the token embedding from the table
        # tok_emb = self.token_embedding_table(idx)

        # I don't understand this part
        # pos_emb = self.position_embedding_table(torch.arange(BLOCK_SIZE, 
                                                            #  device=device))
        # x = tok_emb + pos_emb 
        
        # Get the prediction from the embedding table
        # This is in the structure of [batch_size, block_size, vocab_size]
        # Note that Andrej likes to use time as the dimension to refer to
        # for the block_size because it represents a sequence of tokens
        # that are output over TIME. 
        # So this is (B,T,C)- Batch, Time, Channel
        # logits = self.lm_head(x)

        # If we are doing inferencing, we skip the computation of loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # Flatten the batch and block tensor into a vector (B*T, C)
            logits = logits.view(B*T, C)

            # Flatten the targets the same way (no channel dimension) (B*T)
            targets = targets.view(B*T)

            # Compute the cross entropy loss scalar
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # Iterate over the number of tokens to generate - this is very
        # inefficient as it doesn't do any batch compute whatsoever
        for _ in range(max_new_tokens):
            # So this is (B,T,C)- Batch, Time, Channel
            logits, _ = self(idx)

            # Focus only on the last Time step (last token in input)
            # This yields a (B, C) tensor
            logits = logits[:, -1, :]

            # Compute the probabilities using softmax function
            probs = F.softmax(logits, dim=-1)

            # Compute the next token from probabilities using multinomial
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append the next token to the input
            idx = torch.cat([idx, idx_next], dim=1)

        return idx

# Attribute ensures that pytorch does not compute gradients
@torch.no_grad()
def estimate_loss(m):
    """Estimate the loss on the train and validation sets"""
    out = {}

    # Tell the model to go into inference mode
    m.eval()

    # This loop runs twice - once for training dataset and once for the
    # validation dataset
    for split in ['train', 'val']:

        # Compute the loss for EVAL_ITERS iterations
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):

            # Get a batch of data from either train or validation dataset
            inputs, targets = get_batch(split)

            # Compute the loss for this batch
            _, loss = m(inputs, targets)

            # Store the loss in the losses tensor
            losses[k] = loss.item()

        # Store the mean loss for this dataset
        out[split] = losses.mean()

    # Tell the model to return to training mode
    m.train()

    # Return the losses for train and validation datasets
    return out

# Now let's train the model
m = BigramLanguageModel().to(device)

# Create an optimizer to use for training
optimizer = torch.optim.AdamW(m.parameters(), lr=LEARNING_RATE)

# Training loop
for iter in range(MAX_ITERS):

    # Periodically output the current loss
    if iter % EVAL_INTERAL == 0:
        losses = estimate_loss(m)
        print(f"Step {iter}: Train Loss: {losses['train']:.4f}, "
              f"Val Loss: {losses['val']:.4f}")
    
    # Get batches of data from the training set to use in this iteration
    inputs, targets = get_batch('train')

    # Evaluate the loss
    logits, loss = m(inputs, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# The context is 0, which is a newline character in the vocabulary
context = torch.zeros((1,1), dtype=torch.long, device=device)

# Run the model in inferencing
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))