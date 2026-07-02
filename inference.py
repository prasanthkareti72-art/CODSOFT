import os
import pickle
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import Vocabulary, CNNEncoder, RNNDecoder

class CustomCaptioner:
    """
    Handles inference for the custom ResNet50 + LSTM Image Captioning model.
    """
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.vocabulary = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    def load_model(self, checkpoint_path, vocab_path):
        if not os.path.exists(checkpoint_path) or not os.path.exists(vocab_path):
            raise FileNotFoundError("Model checkpoint or vocabulary file not found. Please train the custom model first.")

        # Load vocabulary
        with open(vocab_path, "rb") as f:
            self.vocabulary = pickle.load(f)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Instantiate models
        self.encoder = CNNEncoder(checkpoint["embed_size"]).to(self.device)
        self.decoder = RNNDecoder(
            checkpoint["embed_size"], 
            checkpoint["hidden_size"], 
            checkpoint["vocab_size"]
        ).to(self.device)

        # Load state weights
        self.encoder.load_state_dict(checkpoint["encoder_state"])
        self.decoder.load_state_dict(checkpoint["decoder_state"])

        self.encoder.eval()
        self.decoder.eval()

    def generate_caption(self, image_path, max_length=20):
        if not self.encoder or not self.decoder or not self.vocabulary:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        image = Image.open(image_path).convert("RGB")
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        caption = []
        with torch.no_grad():
            # Extract image features
            features = self.encoder(img_tensor) # shape: (1, embed_size)
            
            # Step 1: feed image feature vector to LSTM
            lstm_input = features.unsqueeze(1) # shape: (1, 1, embed_size)
            states = None
            lstm_out, states = self.decoder.lstm(lstm_input, states)
            
            # Step 2: feed the start token to begin word generation
            start_idx = self.vocabulary.word2idx["<start>"]
            word_idx = torch.tensor([start_idx]).to(self.device)
            
            for _ in range(max_length):
                # Embed current word index
                embeddings = self.decoder.embed(word_idx).unsqueeze(1) # shape: (1, 1, embed_size)
                
                # Pass into LSTM with past state tracking
                lstm_out, states = self.decoder.lstm(embeddings, states)
                
                # Predict word probabilities
                outputs = self.decoder.linear(lstm_out.squeeze(1)) # shape: (1, vocab_size)
                predicted = outputs.argmax(dim=1)
                
                predicted_item = predicted.item()
                
                # Stop if end token is reached
                if predicted_item == self.vocabulary.word2idx["<end>"]:
                    break
                    
                caption.append(self.vocabulary.idx2word.get(predicted_item, "<unk>"))
                
                # Set predicted index as the next input token
                word_idx = predicted

        return " ".join(caption)


class TransformerCaptioner:
    """
    Handles inference using Hugging Face's SOTA Pretrained ViT-GPT2 Image Captioning Transformer model.
    """
    def __init__(self):
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, status_callback=None):
        def log(msg):
            if status_callback:
                status_callback(msg)
            else:
                print(msg)

        if self.model is not None:
            return # Already loaded

        log("Loading SOTA ViT-GPT2 model... (may take a moment to download on first run)")
        
        from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
        
        model_name = "nlpconnect/vit-gpt2-image-captioning"
        
        log("Downloading ViTImageProcessor...")
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        
        log("Downloading AutoTokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        log("Downloading VisionEncoderDecoderModel...")
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        
        log("Model loaded successfully!")

    def generate_caption(self, image_path):
        if self.model is None:
            raise RuntimeError("Transformer model is not loaded. Call load_model() first.")

        image = Image.open(image_path).convert("RGB")
        
        # Preprocess the image
        pixel_values = self.processor(images=[image], return_tensors="pt").pixel_values.to(self.device)
        
        # Generate prediction ids
        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values, 
                max_length=24, 
                num_beams=4,
                # Avoid repetitive loops in generation
                no_repeat_ngram_size=2
            )
            
        # Decode output ids to get caption
        caption = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return caption.strip()
