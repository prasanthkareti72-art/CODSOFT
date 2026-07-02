import os
import sys
import pickle
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import pyttsx3

# Add current directory to path to ensure local imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train import train
from inference import CustomCaptioner, TransformerCaptioner

class ImageCaptioningApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Captioning AI Studio")
        self.root.geometry("1050x720")
        self.root.minsize(950, 650)
        self.root.configure(bg="#1E1E2E")  # Dark modern background
        
        # Application state
        self.selected_image_path = None
        self.custom_captioner = None
        self.transformer_captioner = None
        self.is_processing = False
        self.is_training = False
        
        # Center the window on the screen
        self.center_window()
        
        # Apply custom styling
        self.setup_styles()
        
        # Draw the layout
        self.create_widgets()
        
        # Log welcome message
        self.write_log("Welcome to Image Captioning AI Studio!")
        self.write_log("Configure settings and select an image to generate a caption.")
        self.write_log("You can use the pre-trained SOTA model or train a custom model from scratch.")

    def center_window(self):
        self.root.update_idletasks()
        width = 1050
        height = 720
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        
        # Configure frames
        style.configure("TFrame", background="#1E1E2E")
        style.configure("Sidebar.TFrame", background="#181825")
        style.configure("Card.TFrame", background="#252538", borderwidth=0)
        
        # Configure labels
        style.configure("TLabel", background="#1E1E2E", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.configure("SidebarHeader.TLabel", background="#181825", foreground="#CBA6F7", font=("Segoe UI", 12, "bold"))
        style.configure("SidebarLabel.TLabel", background="#181825", foreground="#BAC2DE", font=("Segoe UI", 9))
        style.configure("Title.TLabel", background="#1E1E2E", foreground="#F5C2E7", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background="#1E1E2E", foreground="#A6ADC8", font=("Segoe UI", 10))
        style.configure("CardTitle.TLabel", background="#252538", foreground="#F5E0DC", font=("Segoe UI", 11, "bold"))
        style.configure("Caption.TLabel", background="#252538", foreground="#A6E3A1", font=("Segoe UI", 14, "italic", "bold"))
        
        # Configure radio buttons
        style.configure("Sidebar.TRadiobutton", background="#181825", foreground="#CDD6F4", font=("Segoe UI", 10))
        style.map("Sidebar.TRadiobutton", background=[("active", "#181825")], foreground=[("active", "#F5C2E7")])

    def create_widgets(self):
        # 1. Main Grid Layout
        self.root.columnconfigure(0, weight=0, minsize=280)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        # 2. Sidebar Frame
        sidebar = ttk.Frame(self.root, style="Sidebar.TFrame")
        sidebar.grid(row=0, column=0, sticky="nswe", padx=0, pady=0)
        
        # Pad sidebar content
        sidebar.columnconfigure(0, weight=1)
        
        # App Title in Sidebar
        title_label = ttk.Label(sidebar, text="IMAGE CAPTIONER AI", style="SidebarHeader.TLabel")
        title_label.grid(row=0, column=0, sticky="w", padx=20, pady=(25, 5))
        
        desc_label = ttk.Label(sidebar, text="CV & NLP Studio v1.0", style="SidebarLabel.TLabel")
        desc_label.grid(row=1, column=0, sticky="w", padx=20, pady=(0, 20))
        
        # Separator line
        sep1 = ttk.Separator(sidebar, orient="horizontal")
        sep1.grid(row=2, column=0, sticky="ew", padx=15, pady=10)
        
        # Model Selection Label
        model_lbl = ttk.Label(sidebar, text="CHOOSE CAPTIONING MODEL", style="SidebarHeader.TLabel")
        model_lbl.grid(row=3, column=0, sticky="w", padx=20, pady=(15, 10))
        
        self.model_var = tk.StringVar(value="transformer")
        
        rb1 = ttk.Radiobutton(
            sidebar, 
            text="SOTA Transformer (ViT-GPT2)", 
            variable=self.model_var, 
            value="transformer", 
            style="Sidebar.TRadiobutton",
            command=self.on_model_change
        )
        rb1.grid(row=4, column=0, sticky="w", padx=30, pady=5)
        
        rb2 = ttk.Radiobutton(
            sidebar, 
            text="Custom CNN-LSTM (ResNet)", 
            variable=self.model_var, 
            value="custom", 
            style="Sidebar.TRadiobutton",
            command=self.on_model_change
        )
        rb2.grid(row=5, column=0, sticky="w", padx=30, pady=5)
        
        # Separator 2
        sep2 = ttk.Separator(sidebar, orient="horizontal")
        sep2.grid(row=6, column=0, sticky="ew", padx=15, pady=20)
        
        # Custom Training Configuration Label
        train_lbl = ttk.Label(sidebar, text="TRAIN CUSTOM MODEL", style="SidebarHeader.TLabel")
        train_lbl.grid(row=7, column=0, sticky="w", padx=20, pady=(5, 10))
        
        # Epochs Slider
        epochs_lbl = ttk.Label(sidebar, text="Training Epochs:", style="SidebarLabel.TLabel")
        epochs_lbl.grid(row=8, column=0, sticky="w", padx=25, pady=(5, 2))
        
        self.epochs_var = tk.IntVar(value=15)
        self.epochs_scale = tk.Scale(
            sidebar, from_=5, to=50, orient="horizontal", 
            variable=self.epochs_var, bg="#181825", fg="#CDD6F4", 
            troughcolor="#11111B", highlightbackground="#181825", 
            activebackground="#CBA6F7", bd=0, font=("Segoe UI", 9)
        )
        self.epochs_scale.grid(row=9, column=0, sticky="ew", padx=25, pady=(0, 10))
        
        # Batch Size Slider
        batch_lbl = ttk.Label(sidebar, text="Batch Size:", style="SidebarLabel.TLabel")
        batch_lbl.grid(row=10, column=0, sticky="w", padx=25, pady=(5, 2))
        
        self.batch_var = tk.IntVar(value=4)
        self.batch_scale = tk.Scale(
            sidebar, from_=2, to=32, orient="horizontal", 
            variable=self.batch_var, bg="#181825", fg="#CDD6F4", 
            troughcolor="#11111B", highlightbackground="#181825", 
            activebackground="#CBA6F7", bd=0, font=("Segoe UI", 9)
        )
        self.batch_scale.grid(row=11, column=0, sticky="ew", padx=25, pady=(0, 15))
        
        # Train Button (Flat modern style)
        self.btn_train = tk.Button(
            sidebar, text="⚡ Run Training Loop", 
            bg="#CBA6F7", fg="#11111B", font=("Segoe UI", 10, "bold"),
            relief="flat", activebackground="#B4BEFE", activeforeground="#11111B",
            cursor="hand2", command=self.start_training_thread
        )
        self.btn_train.grid(row=12, column=0, sticky="ew", padx=20, pady=10)
        self.make_hoverable(self.btn_train, "#B4BEFE", "#CBA6F7")
        
        # Footer
        footer = ttk.Label(sidebar, text="Uses PyTorch & Hugging Face", style="SidebarLabel.TLabel")
        footer.grid(row=13, column=0, sticky="s", pady=(50, 15))

        # 3. Main Content Panel
        main_panel = ttk.Frame(self.root, style="TFrame")
        main_panel.grid(row=0, column=1, sticky="nswe", padx=25, pady=20)
        
        main_panel.columnconfigure(0, weight=1)
        main_panel.rowconfigure(1, weight=1) # Image container takes space
        
        # Header text
        header_frame = ttk.Frame(main_panel, style="TFrame")
        header_frame.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        
        header_lbl = ttk.Label(header_frame, text="Image Captioning AI Studio", style="Title.TLabel")
        header_lbl.pack(anchor="w")
        sub_lbl = ttk.Label(header_frame, text="Upload an image to see the caption generated by neural network", style="Subtitle.TLabel")
        sub_lbl.pack(anchor="w", pady=(2, 0))
        
        # Image Display Area (Card layout)
        self.image_card = ttk.Frame(main_panel, style="Card.TFrame")
        self.image_card.grid(row=1, column=0, sticky="nswe", pady=(0, 15))
        
        self.image_card.columnconfigure(0, weight=1)
        self.image_card.rowconfigure(0, weight=1)
        
        # Placeholder inside Image Card
        self.img_display_label = tk.Label(
            self.image_card, text="📸 Click to load an image", 
            bg="#252538", fg="#A6ADC8", font=("Segoe UI", 12, "bold"),
            cursor="hand2"
        )
        self.img_display_label.grid(row=0, column=0, sticky="nswe")
        self.img_display_label.bind("<Button-1>", lambda e: self.select_image())
        
        # Caption Action Button & Model indicator
        btn_frame = ttk.Frame(main_panel, style="TFrame")
        btn_frame.grid(row=2, column=0, sticky="ew", pady=(0, 15))
        btn_frame.columnconfigure(0, weight=1)
        btn_frame.columnconfigure(1, weight=0)
        
        self.btn_caption = tk.Button(
            btn_frame, text="✨ Generate Caption", 
            bg="#89B4FA", fg="#11111B", font=("Segoe UI", 12, "bold"),
            relief="flat", activebackground="#A6E3A1", activeforeground="#11111B",
            cursor="hand2", command=self.generate_caption_thread, height=2
        )
        self.btn_caption.grid(row=0, column=0, sticky="ew", padx=(0, 10))
        self.make_hoverable(self.btn_caption, "#B4BEFE", "#89B4FA")
        
        self.btn_load_image = tk.Button(
            btn_frame, text="📂 Load File", 
            bg="#313244", fg="#CDD6F4", font=("Segoe UI", 12),
            relief="flat", activebackground="#45475A", activeforeground="#CDD6F4",
            cursor="hand2", command=self.select_image, width=12, height=2
        )
        self.btn_load_image.grid(row=0, column=1, sticky="e")
        self.make_hoverable(self.btn_load_image, "#45475A", "#313244")

        # Caption Result Card
        self.result_card = ttk.Frame(main_panel, style="Card.TFrame")
        self.result_card.grid(row=3, column=0, sticky="ew", pady=(0, 15))
        
        self.result_card.columnconfigure(0, weight=1)
        self.result_card.columnconfigure(1, weight=0)
        
        res_header = ttk.Label(self.result_card, text="GENERATED CAPTION", style="CardTitle.TLabel")
        res_header.grid(row=0, column=0, sticky="w", padx=20, pady=(15, 5))
        
        self.caption_text_label = ttk.Label(
            self.result_card, text="Select an image and click Generate.", 
            style="Caption.TLabel", wraplength=600
        )
        self.caption_text_label.grid(row=1, column=0, sticky="w", padx=20, pady=(5, 20))
        
        self.btn_speak = tk.Button(
            self.result_card, text="🔊 Listen", 
            bg="#45475A", fg="#A6E3A1", font=("Segoe UI", 9, "bold"),
            relief="flat", cursor="hand2", command=self.on_speak_click, state="disabled"
        )
        self.btn_speak.grid(row=0, column=1, rowspan=2, sticky="e", padx=25, pady=20)
        self.make_hoverable(self.btn_speak, "#585B70", "#45475A")

        # Console Log Terminal (scrolled log box)
        console_frame = ttk.Frame(main_panel, style="TFrame")
        console_frame.grid(row=4, column=0, sticky="ew")
        
        console_hdr = ttk.Label(console_frame, text="LOG CONSOLE", style="SidebarHeader.TLabel")
        console_hdr.pack(anchor="w", pady=(0, 5))
        
        self.console_log = tk.Text(
            console_frame, height=7, bg="#11111B", fg="#A6E3A1", 
            font=("Consolas", 9), relief="flat", bd=0, padx=10, pady=8
        )
        self.console_log.pack(fill="x", expand=True)
        
        scrollbar = ttk.Scrollbar(self.console_log, command=self.console_log.yview)
        # We overlay scrollbar slightly to keep custom clean design
        scrollbar.pack(side="right", fill="y")
        self.console_log.config(yscrollcommand=scrollbar.set)
        self.console_log.config(state="disabled")

    def make_hoverable(self, button, hover_bg, normal_bg):
        button.bind("<Enter>", lambda e: button.config(bg=hover_bg) if button["state"] != "disabled" else None)
        button.bind("<Leave>", lambda e: button.config(bg=normal_bg) if button["state"] != "disabled" else None)

    def write_log(self, message):
        """Append a line to the console log, maintaining scroll."""
        self.console_log.config(state="normal")
        self.console_log.insert("end", f"> {message}\n")
        self.console_log.see("end")
        self.console_log.config(state="disabled")
        self.root.update_idletasks()

    def select_image(self):
        if self.is_processing or self.is_training:
            return
            
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if not file_path:
            return
            
        self.selected_image_path = file_path
        self.write_log(f"Loaded image: {os.path.basename(file_path)}")
        
        # Load and render image fitted to container
        img = Image.open(file_path)
        
        # Get dimensions and resize keeping aspect ratio
        max_w = self.image_card.winfo_width()
        max_h = self.image_card.winfo_height()
        if max_w < 100: max_w = 600
        if max_h < 100: max_h = 300
        
        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        
        self.tk_image = ImageTk.PhotoImage(img)
        self.img_display_label.config(image=self.tk_image, text="")
        self.caption_text_label.config(text="Ready to generate caption.")
        self.btn_speak.config(state="disabled")

    def on_model_change(self):
        model_type = self.model_var.get()
        self.write_log(f"Switched model mode to: {model_type.upper()}")

    def on_speak_click(self):
        caption = self.caption_text_label.cget("text")
        if caption and caption not in ["Ready to generate caption.", "Select an image and click Generate."]:
            self.write_log("Speaking caption...")
            def run_speech():
                try:
                    engine = pyttsx3.init()
                    engine.say(caption)
                    engine.runAndWait()
                except Exception as e:
                    self.write_log(f"TTS Speech error: {e}")
            threading.Thread(target=run_speech, daemon=True).start()

    def generate_caption_thread(self):
        if self.is_processing:
            return
        if not self.selected_image_path:
            messagebox.showwarning("No Image", "Please load an image first!")
            return
            
        self.is_processing = True
        self.btn_caption.config(state="disabled", bg="#585B70", text="⌛ Generating...")
        self.btn_load_image.config(state="disabled")
        self.btn_train.config(state="disabled")
        self.btn_speak.config(state="disabled")
        
        # Launch background captioning thread
        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        model_type = self.model_var.get()
        caption = ""
        
        try:
            if model_type == "transformer":
                if not self.transformer_captioner:
                    self.transformer_captioner = TransformerCaptioner()
                    self.transformer_captioner.load_model(status_callback=self.write_log)
                
                self.write_log("Generating SOTA caption...")
                caption = self.transformer_captioner.generate_caption(self.selected_image_path)
                
            else: # Custom RNN model
                if not self.custom_captioner:
                    self.custom_captioner = CustomCaptioner()
                    
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    chk_path = os.path.join(current_dir, "checkpoint.pth")
                    vocab_path = os.path.join(current_dir, "vocab.pkl")
                    
                    self.custom_captioner.load_model(chk_path, vocab_path)
                    
                self.write_log("Generating custom model caption...")
                caption = self.custom_captioner.generate_caption(self.selected_image_path)
                
            self.write_log(f"Caption generated: '{caption}'")
            # Apply typewriter effect in main thread
            self.root.after(0, self.animate_caption_output, caption)
            
        except Exception as e:
            self.write_log(f"Error: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Caption Generation Error", str(e)))
            self.root.after(0, lambda: self.caption_text_label.config(text="Generation failed."))
            
        finally:
            self.root.after(0, self.reset_inference_buttons)

    def animate_caption_output(self, caption):
        self.btn_speak.config(state="normal")
        self.typewrite_text(self.caption_text_label, caption, 0)

    def typewrite_text(self, label, text, index=0):
        if index < len(text):
            label.config(text=text[:index+1])
            # Set speed of typewriter typing
            label.after(40, self.typewrite_text, label, text, index+1)

    def reset_inference_buttons(self):
        self.is_processing = False
        self.btn_caption.config(state="normal", bg="#89B4FA", text="✨ Generate Caption")
        self.btn_load_image.config(state="normal")
        self.btn_train.config(state="normal")

    def start_training_thread(self):
        if self.is_training or self.is_processing:
            return
            
        confirm = messagebox.askyesno(
            "Start Training", 
            "This will generate a synthetic toy dataset (or use an existing one) "
            "and run PyTorch training for the Custom ResNet-LSTM model. Start training?"
        )
        if not confirm:
            return
            
        self.is_training = True
        self.btn_train.config(state="disabled", bg="#585B70", text="⚙️ Training...")
        self.btn_caption.config(state="disabled")
        self.btn_load_image.config(state="disabled")
        
        epochs = self.epochs_var.get()
        batch_size = self.batch_var.get()
        
        # Clear custom captioner instance to force reload new weights after training
        self.custom_captioner = None
        
        threading.Thread(
            target=self.run_training, 
            args=(epochs, batch_size), 
            daemon=True
        ).start()

    def run_training(self, epochs, batch_size):
        try:
            self.write_log(f"Initiating training: {epochs} epochs, batch size {batch_size}")
            
            # Pass our write_log method directly as log callback to write to the GUI console
            train(epochs=epochs, batch_size=batch_size, log_callback=self.write_log)
            
            self.write_log("Training completed successfully! Model weights saved as checkpoint.pth.")
            self.root.after(0, lambda: messagebox.showinfo("Training Success", "Custom model successfully trained and weights saved!"))
        except Exception as e:
            self.write_log(f"Training Failed: {str(e)}")
            self.root.after(0, lambda: messagebox.showerror("Training Error", f"Training failed:\n{str(e)}"))
        finally:
            self.root.after(0, self.reset_training_buttons)

    def reset_training_buttons(self):
        self.is_training = False
        self.btn_train.config(state="normal", bg="#CBA6F7", text="⚡ Run Training Loop")
        self.btn_caption.config(state="normal")
        self.btn_load_image.config(state="normal")

if __name__ == "__main__":
    print("1. Starting app")

    root = tk.Tk()
    print("2. Tk window created")

    app = ImageCaptioningApp(root)
    print("3. GUI created")

    root.mainloop()
    print("4. Mainloop exited")
