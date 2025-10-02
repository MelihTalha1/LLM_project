import os
import gradio as gr
import torch
from llm1.llm_model import UstaModel
from llm1.llm_tokenizer import UstaTokenizer

# Load the model and tokenizer
def load_model(custom_model_path=None):
    try:
        llm_tokenizer = UstaTokenizer("llm1/tokenizer.json")
        print("✅ Tokenizer loaded successfully! vocab size:", len(llm_tokenizer.vocab))
        
        # Model parameters - adjust these to match your trained model
        context_length = 64
        vocab_size = len(llm_tokenizer.vocab)
        embedding_dim = 128
        num_heads = 8
        num_layers = 8
        device = "cpu"  # Use CPU for compatibility
        
        # Load the model
        llm_model = UstaModel(
            vocab_size=vocab_size, 
            embedding_dim=embedding_dim,
            num_heads=num_heads, 
            context_length=context_length, 
            num_layers=num_layers,
            device=device
        )        
        # Determine which model file to use
        if custom_model_path and os.path.exists(custom_model_path):
            model_path = custom_model_path
            print(f"🎯 Using uploaded model: {model_path}")
        else:
            model_path = "llm1/llm_model.pth"
            
            if not os.path.exists(model_path):
                print("❌ Model file not found at", model_path)
                # Download the model file from GitHub
                try:
                    print("📥 Downloading model weights from GitHub...")
                    import requests
                    url = "https://github.com/..."
                    
                    headers = {
                        'Accept': 'application/octet-stream',
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    }
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()  # Raise an exception for bad status codes
                    
                    # Check if we got a proper binary file (PyTorch files start with specific bytes)
                    if response.content[:4] != b'PK\x03\x04' and b'<html' in response.content[:100].lower():
                        raise Exception("Downloaded HTML instead of binary file - check URL")
                    
                    print(f"📦 Downloaded {len(response.content)} bytes")
                    
                    # Create llm1 directory if it doesn't exist
                    os.makedirs("llm1", exist_ok=True)
                    
                    # Save the model weights to the local file system
                    with open(model_path, "wb") as f:
                        f.write(response.content)
                    print("✅ Model weights saved successfully!")
                except Exception as e:
                    print(f"❌ Failed to download model weights: {e}")
                    print("Using random initialization.")

        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                
                # Handle potential key mapping issues
                if "embedding.weight" in state_dict and "embedding.embedding.weight" not in state_dict:
                    # Map old key names to new key names
                    new_state_dict = {}
                    for key, value in state_dict.items():
                        if key == "embedding.weight":
                            new_state_dict["embedding.embedding.weight"] = value
                        elif key == "pos_embedding.weight":
                            # Skip positional embedding if not expected
                            continue
                        else:
                            new_state_dict[key] = value
                    state_dict = new_state_dict
                
                llm_model.load_state_dict(state_dict)
                llm_model.eval()
                print("✅ Model weights loaded successfully!")
                return llm_model, llm_tokenizer, f"✅ Model loaded from: {model_path}"
            except Exception as e:
                print(f"⚠️ Warning: Could not load trained weights: {e}")
                print("Using random initialization.")
                return llm_model, llm_tokenizer, f"⚠️ Failed to load weights: {e}"
        else:
            print(f"⚠️ Model file not found at {model_path}. Using random initialization.")
            return llm_model, llm_tokenizer, "⚠️ Using random initialization"
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e

# Global model variables
model, tokenizer, model_status = None, None, "Not loaded"

# Initialize model and tokenizer globally
try:
    model, tokenizer, model_status = load_model()
    print("🚀 LlmModel and tokenizer initialized successfully!")
except Exception as e:
    print(f"❌ Failed to initialize model: {e}")
    model, tokenizer, model_status = None, None, f"❌ Error: {e}"

def load_model_from_url(url):
    """Load model from a URL"""
    global model, tokenizer, model_status
    
    if not url.strip():
        return "❌ Please provide a URL"
    
    try:
        print(f"📥 Downloading model from URL: {url}")
        import requests
        
        headers = {
            'Accept': 'application/octet-stream',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Check if we got a proper binary file
        if response.content[:4] != b'PK\x03\x04' and b'<html' in response.content[:100].lower():
            return "❌ Downloaded HTML instead of binary file - check URL"
        
        # Save temporary file
        temp_path = "temp_model.pth"
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        # Load the model
        new_model, new_tokenizer, status = load_model(temp_path)
        
        # Update global variables
        model = new_model
        tokenizer = new_tokenizer
        model_status = status
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return status
    except Exception as e:
        error_msg = f"❌ Failed to load model from URL: {e}"
        model_status = error_msg
        return error_msg

def load_model_from_file(uploaded_file):
    """Load model from uploaded file"""
    global model, tokenizer, model_status
    
    if uploaded_file is None:
        return "❌ No file uploaded"
    
    try:
        # Check if the file path exists and is valid
        file_path = uploaded_file.name if hasattr(uploaded_file, 'name') else str(uploaded_file)
        
        # For HF Spaces compatibility, also try the upload path
        if not os.path.exists(file_path) and hasattr(uploaded_file, 'orig_name'):
            # Sometimes HF Spaces provides different paths
            print(f"Original path not found: {file_path}")
            print(f"Trying original name: {uploaded_file.orig_name}")
            file_path = uploaded_file.orig_name
        
        print(f"📁 Attempting to load model from: {file_path}")
        
        # Load the new model
        new_model, new_tokenizer, status = load_model(file_path)
        
        # Update global variables
        model = new_model
        tokenizer = new_tokenizer
        model_status = status
        
        return status
    except Exception as e:
        error_msg = f"❌ Failed to load uploaded model: {e}"
        print(f"Error details: {e}")
        model_status = error_msg
        return error_msg

def chat_with_usta(message, history, max_tokens=20, temperature=1.0, top_k=64, top_p=1.0):
    """Simple chat function"""
    if model is None or tokenizer is None:
        return history + [["Error", "UstaModel is not available. Please try again later."]]
        
    try:
        # Encode the input message
        tokens = tokenizer.encode(message)
        
        # Make sure we don't exceed context length
        if len(tokens) > 25:  # Leave some room for generation
            tokens = tokens[-25:]
        
        # Generate response
        with torch.no_grad():
            actual_max_tokens = min(max_tokens, 32 - len(tokens))
            generated_tokens = model.generate(
                tokens, 
                max_new_tokens=actual_max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )
        
        # Decode the generated tokens
        response = tokenizer.decode(generated_tokens)
        
        # Clean up the response (remove the original input)
        original_text = tokenizer.decode(tokens.tolist())
        if response.startswith(original_text):
            response = response[len(original_text):]
        
        # Clean up any unwanted tokens
        response = response.replace("[UNK]", "").replace("[PAD]", "").strip()
        
        if not response:
            response = "I'm not sure how to respond to that with my knowledge."
            
        # Add to history
        history.append([message, response])
        return history
        
    except Exception as e:
        history.append([message, f"Sorry, I encountered an error: {str(e)}"])
        return history

# Create simple interface
with gr.Blocks(title="🤖 Llm Model Chat") as demo:
    gr.Markdown("# 🤖 Llm Model Chat")
    gr.Markdown("Chat with a custom transformer language model built from scratch! This model specializes in geographical knowledge.")
    
    # Simple chat interface
    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(label="Your message", placeholder="Ask about countries, capitals, or cities...")
    
    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear")
    
    # Generation settings
    gr.Markdown("## ⚙️ Generation Settings")
    with gr.Row():
        max_tokens = gr.Slider(minimum=1, maximum=30, value=20, step=1, label="Max tokens")
        temperature = gr.Slider(minimum=0.1, maximum=2.0, value=1.0, step=0.1, label="Temperature")
    
    with gr.Row():
        top_k = gr.Slider(minimum=1, maximum=64, value=40, step=1, label="Top-k")
        top_p = gr.Slider(minimum=0.1, maximum=1.0, value=1.0, step=0.05, label="Top-p (nucleus sampling)")
    
    # Model loading (simplified)
    gr.Markdown("## 🔧 Load Custom Model")
    with gr.Row():
        model_url = gr.Textbox(
            label="Model URL", 
            placeholder="https://github.com/...",
            scale=3
        )
        load_url_btn = gr.Button("Load from URL", scale=1)
    
    with gr.Row():
        model_file = gr.File(label="Upload model file (.pth, .pt, .bin)")
        load_file_btn = gr.Button("Load File", scale=1)
    
    status = gr.Textbox(label="Status", value=model_status, interactive=False)
    
    # Event handlers
    def send_message(message, history, max_tok, temp, k, p):
        if not message.strip():
            return history, ""
        return chat_with_usta(message, history, max_tok, temp, k, p), ""
    
    send_btn.click(
        send_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_k, top_p],
        outputs=[chatbot, msg]
    )
    
    msg.submit(
        send_message,
        inputs=[msg, chatbot, max_tokens, temperature, top_k, top_p],
        outputs=[chatbot, msg]
    )
    
    clear_btn.click(lambda: [], outputs=[chatbot])
    
    load_url_btn.click(
        load_model_from_url,
        inputs=[model_url],
        outputs=[status]
    )
    
    load_file_btn.click(
        load_model_from_file,
        inputs=[model_file],
        outputs=[status]
    )

if __name__ == "__main__":
    demo.launch()
