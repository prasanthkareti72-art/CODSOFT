import os
import pickle
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
from model import Vocabulary, CNNEncoder, RNNDecoder

class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, captions_file, vocabulary, transform=None):
        self.root_dir = root_dir
        # Load captions file (expects columns: image, caption)
        self.df = pd.read_csv(captions_file)
        self.vocabulary = vocabulary
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.iloc[idx]['caption']
        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.root_dir, img_name)
        
        # Load image
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Numericalize caption
        numericalized_caption = [self.vocabulary.word2idx["<start>"]]
        numericalized_caption.extend(self.vocabulary.numericalize(caption))
        numericalized_caption.append(self.vocabulary.word2idx["<end>"])
        
        return image, torch.tensor(numericalized_caption)


class CollateCaps:
    """
    Collate function to pad captions in a batch to the same length.
    """
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images = [item[0].unsqueeze(0) for item in batch]
        images = torch.cat(images, dim=0)
        
        targets = [item[1] for item in batch]
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        
        return images, targets


def generate_toy_dataset(output_dir):
    """
    Generates a small dataset of synthetic images and captions for testing/demo purposes.
    """
    print("Generating synthetic toy dataset...")
    images_dir = os.path.join(output_dir, "toy_images")
    os.makedirs(images_dir, exist_ok=True)
    
    captions_file = os.path.join(output_dir, "toy_captions.csv")
    
    data = []
    shapes = [
        ("circle", "red", (255, 0, 0)),
        ("square", "blue", (0, 0, 255)),
        ("triangle", "green", (0, 255, 0)),
        ("circle", "yellow", (255, 255, 0)),
        ("square", "purple", (128, 0, 128)),
        ("triangle", "orange", (255, 165, 0)),
    ]
    
    backgrounds = [
        ("black", (0, 0, 0)),
        ("white", (255, 255, 255)),
    ]
    
    counter = 1
    for shape, color_name, color_rgb in shapes:
        for bg_name, bg_rgb in backgrounds:
            # Create a 256x256 image
            img = Image.new('RGB', (256, 256), color=bg_rgb)
            draw = ImageDraw.Draw(img)
            
            # Draw shape in the center
            if shape == "circle":
                draw.ellipse([64, 64, 192, 192], fill=color_rgb)
            elif shape == "square":
                draw.rectangle([64, 64, 192, 192], fill=color_rgb)
            elif shape == "triangle":
                draw.polygon([(128, 64), (64, 192), (192, 192)], fill=color_rgb)
                
            img_filename = f"image_{counter}.jpg"
            img.save(os.path.join(images_dir, img_filename))
            
            # Caption format
            caption = f"a {color_name} {shape} on a {bg_name} background"
            data.append({"image": img_filename, "caption": caption})
            counter += 1
            
    df = pd.DataFrame(data)
    df.to_csv(captions_file, index=False)
    print(f"Generated {len(data)} toy images and captions at: {output_dir}")
    return images_dir, captions_file


def train(data_dir=None, epochs=10, batch_size=4, lr=3e-4, embed_size=256, hidden_size=256, log_callback=None):
    """
    Main training function. Supports custom callbacks for UI integration.
    """
    def log(message):
        if log_callback:
            log_callback(message)
        else:
            print(message)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device}")
    
    # Path setup
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    if data_dir is None:
        data_dir = os.path.join(current_dir, "toy_data")
        
    images_dir = os.path.join(data_dir, "toy_images")
    captions_file = os.path.join(data_dir, "toy_captions.csv")
    
    # Generate toy data if it doesn't exist
    if not os.path.exists(images_dir) or not os.path.exists(captions_file):
        images_dir, captions_file = generate_toy_dataset(data_dir)

    # Load captions to build vocabulary
    df = pd.read_csv(captions_file)
    vocabulary = Vocabulary(freq_threshold=1)
    vocabulary.build_vocabulary(df['caption'].tolist())
    
    vocab_size = len(vocabulary)
    log(f"Vocabulary size: {vocab_size}")

    # Save vocabulary
    vocab_path = os.path.join(current_dir, "vocab.pkl")
    with open(vocab_path, "wb") as f:
        pickle.dump(vocabulary, f)
    log(f"Saved vocabulary to {vocab_path}")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    # Dataset & Dataloader
    dataset = ImageCaptionDataset(images_dir, captions_file, vocabulary, transform=transform)
    pad_idx = vocabulary.word2idx["<pad>"]
    
    # Batch size safety check for toy dataset
    if len(dataset) < batch_size:
        batch_size = len(dataset)
        
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=CollateCaps(pad_idx)
    )

    # Initialize Models
    encoder = CNNEncoder(embed_size).to(device)
    decoder = RNNDecoder(embed_size, hidden_size, vocab_size).to(device)
    
    # Optimizer and Loss
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # We only train encoder's FC classification layer and decoder's weights
    params = list(decoder.parameters()) + list(encoder.resnet.fc.parameters())
    optimizer = optim.Adam(params, lr=lr)

    log("Starting training...")
    encoder.train()
    decoder.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        
        for idx, (images, captions) in enumerate(dataloader):
            images = images.to(device)
            captions = captions.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = decoder(features, captions)
            
            # Reshape outputs to calculate loss
            # outputs shape: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
            # targets shape: (batch_size, seq_len) -> (batch_size * seq_len)
            loss = criterion(
                outputs.view(-1, outputs.shape[2]), 
                captions.view(-1)
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        log(f"Epoch [{epoch}/{epochs}] - Loss: {avg_loss:.4f}")
        
    # Save checkpoint
    checkpoint = {
        "encoder_state": encoder.state_dict(),
        "decoder_state": decoder.state_dict(),
        "embed_size": embed_size,
        "hidden_size": hidden_size,
        "vocab_size": vocab_size
    }
    
    checkpoint_path = os.path.join(current_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)
    log(f"Saved model checkpoint to {checkpoint_path}")
    log("Training complete successfully!")

if __name__ == "__main__":
    train(epochs=15, batch_size=4)
