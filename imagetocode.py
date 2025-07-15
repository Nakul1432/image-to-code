import torch


from datasets import load_dataset
dataset = load_dataset("HuggingFaceM4/WebSight", name="v0.1")
train = dataset["train"]


import random
i= random.sample(range(len(train)), 1000)
train= train.select(i)



 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from PIL import Image
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence


def tokenize(text):
    return text.lower().split()

def build_vocab(dataset, min_freq=1):
    counter = Counter()
    for item in dataset:
        tokens = tokenize(item["text"])
        counter.update(tokens)

    specials = ["<pad>", "<sos>", "<eos>", "<unk>"]
    vocab = {token: idx for idx, token in enumerate(specials)}
    idx = len(vocab)

    for token, freq in counter.items():
        if freq >= min_freq:
            vocab[token] = idx
            idx += 1

    return vocab

def encode(text, vocab, max_length=512):
    tokens = ["<sos>"] + tokenize(text)[:max_length - 2] + ["<eos>"]
    return [vocab.get(token, vocab["<unk>"]) for token in tokens]


class CustomDataset(Dataset):
    def __init__(self, data, vocab, max_length=5120):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = self.transform(item["image"].convert("RGB"))
        code_ids = torch.tensor(encode(item["text"], self.vocab, self.max_length), dtype=torch.long)
        return image, code_ids


class Encoder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.cnn(x).view(x.size(0), -1)
        return x.unsqueeze(1) 

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads=4, num_layers=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos = nn.Parameter(torch.randn(1, 512, embed_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=num_heads)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt, memory):
    
     tgt = self.embed(tgt) + self.pos[:, :tgt.size(1)]

    
     tgt = tgt.transpose(0, 1)
     memory = memory.transpose(0, 1)

     tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

     out = self.transformer(tgt, memory, tgt_mask=tgt_mask)

     out = out.transpose(0, 1)
     return self.fc(out)


class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim=256):
        super().__init__()
        self.encoder = Encoder(embed_dim)
        self.decoder = Decoder(vocab_size, embed_dim)

    def forward(self, img, tgt):
        memory = self.encoder(img)
        return self.decoder(tgt, memory)


def collate_fn(batch):
    imgs, codes = zip(*batch)
    imgs = torch.stack(imgs)
    codes_in = [c[:-1] for c in codes]
    codes_out = [c[1:] for c in codes]
    codes_in = pad_sequence(codes_in, batch_first=True, padding_value=vocab["<pad>"])
    codes_out = pad_sequence(codes_out, batch_first=True, padding_value=vocab["<pad>"])
    return imgs, codes_in, codes_out


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for img, codes_in, codes_out in dataloader:
        img, codes_in, codes_out = img.to(device), codes_in.to(device), codes_out.to(device)
        out = model(img, codes_in)
        out = out.view(-1, out.size(-1))
        loss = loss_fn(out, codes_out.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)


your_dataset = []
for item in train:
    image = item["image"]
    image.load()
    your_dataset.append({
        "image": image,
        "text": item["text"]
    })



import pickle
import os


if os.path.exists("vocab.pkl"):
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    print("existing vocab.")
else:
    vocab = build_vocab(your_dataset)
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print(" new vocab.")



dataset = CustomDataset(your_dataset, vocab)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(vocab_size=len(vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab["<pad>"])

import os

start_epoch = 0
best_loss = float("inf")  
if os.path.exists("best_checkpoint.pth"):
    checkpoint = torch.load("best_checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint["loss"]
    print(f"Resuming : epoch {start_epoch} | loss {best_loss:.4f}")
else:
    print("Starting from scratch.")




for epoch in range(25):
    loss = train_one_epoch(model, dataloader, optimizer, loss_fn, device=device)
    print(f"Epoch {epoch+1}: Loss = {loss:.4f}")

    
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, "checkpoint.pth")


if loss < best_loss:
    best_loss = loss
    torch.save(checkpoint, "best_checkpoint.pth")
    print(f"Saved new model at epoch {epoch+1} with loss {loss:.4f}")

    torch.save(checkpoint, "checkpoint.pth")



#inference
def test(model, image_tensor, vocab, max_len=100):
    model.eval()
    inv_vocab = {v: k for k, v in vocab.items()}
    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]

    with torch.no_grad():
        memory = model.encoder(image_tensor.unsqueeze(0).to(device))
        output_ids = [sos_id]

        for _ in range(max_len):
            tgt = torch.tensor(output_ids, dtype=torch.long).unsqueeze(0).to(device)
            out = model.decoder(tgt, memory)
            next_token_id = out[0, -1].argmax(dim=-1).item()
            output_ids.append(next_token_id)
            if next_token_id == eos_id:
                break

        decoded = [inv_vocab.get(i, "<unk>") for i in output_ids]
        return " ".join(decoded[1:-1])


#test_img, test_code = dataset[0]   
#generated_code = test(model, test_img, vocab)


#print("\nPredicted:\n", generated_code)

import pickle
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)


checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'loss': loss
}
torch.save(checkpoint, "checkpoint.pth")

