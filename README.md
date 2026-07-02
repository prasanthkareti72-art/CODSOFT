# Image Captioning AI Studio

A standalone Python desktop application combining **Computer Vision (CV)** and **Natural Language Processing (NLP)** to automatically generate text captions for images. It features both a custom-trained Deep Learning model and a State-of-the-Art (SOTA) Transformer model.

---

## Features

1. **Custom CNN-LSTM Encoder-Decoder Model**:
   - **Encoder**: Pre-trained ResNet-50 network that extracts high-level semantic visual features.
   - **Decoder**: LSTM neural network that models word dependencies and generates captions sequentially.
2. **SOTA Pretrained Transformer**:
   - Out-of-the-box integration with a Vision-Encoder-Decoder model (`nlp-connect/vit-gpt2-image-captioning`) from Hugging Face for high-accuracy production results.
3. **Interactive Dark GUI**:
   - Modern, stylish, and responsive UI built using Python `tkinter`.
   - Thread-safe background jobs to prevent interface freezes during training and captioning.
   - Live log console showing real-time training progress, training loss, and downloads.
   - Built-in **Text-to-Speech (TTS)** engine that reads generated captions aloud.
4. **Synthetic Dataset Generator**:
   - Generates a local synthetic toy dataset of shapes (circles, squares, triangles) on colored backgrounds to allow running and validating the custom model training loop instantly.

---

## File Structure

- [model.py](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/model.py): Deep Learning architectures (`Vocabulary`, `CNNEncoder`, `RNNDecoder`) in PyTorch.
- [train.py](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/train.py): Dataset loaders, collator, synthetic data generator, and optimizer loops.
- [inference.py](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/inference.py): Prediction routines for the Custom model and the SOTA model.
- [app.py](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/app.py): Entrypoint file containing the Tkinter interface code.
- [run.bat](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/run.bat): One-click executable file for Windows users.

---

## Installation & Setup

All required packages (`torch`, `torchvision`, `transformers`, `pillow`, `pandas`, `pyttsx3`) are already pre-installed in your virtual/system environment. 

To start the application:
1. Double-click [run.bat](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/run.bat) (or run `python app.py` from the command line in this folder).
2. Choose **SOTA Transformer** for high quality immediate predictions. When you generate a caption for the first time, it will automatically download the pretrained weights from Hugging Face.
3. Choose **Custom CNN-LSTM (ResNet)** to test the custom model. Click **Run Training Loop** on the sidebar to train it first (it creates synthetic images and trains them in a few seconds on a CPU/GPU).

---

## Training on a Custom Dataset (e.g., Flickr8k)

To train on your own real dataset:
1. Organize your images in a folder (e.g., `images/`).
2. Create a CSV file (e.g., `captions.csv`) with two columns:
   - `image`: The name of the file (e.g., `1000268201_693b08cb0e.jpg`).
   - `caption`: The text description (e.g., `A child playing in a pool.`).
3. Call the `train` function in [train.py](file:///c:/Users/mail2/OneDrive/Desktop/BPY_CSD_2580-WATER%20SCARCITY%20MANAGEMENT%20THROUGH%20CENTRALIZED%20KNOWLEDGE-SHARING%20PLATFORM/SOURCE%20CODE/Water%20Scarcity%20Management%20through%20Centralized%20Knowledge-Sharing%20Platform/ImageCaptioning/train.py) pointing to your directories:
   ```python
   from train import train
   
   train(
       data_dir="path/to/dataset_folder", # containing images/ and captions.csv
       epochs=20, 
       batch_size=32
   )
   ```
4. Once completed, copy the generated `checkpoint.pth` and `vocab.pkl` into the main application directory to load them in the UI!
