from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

MODEL_ID = "trocr_model_original"  # o trocr-large-handwritten / trocr-base-printed

def load():
    processor = TrOCRProcessor.from_pretrained(MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)
    model.eval()

    # Verifica que los token IDs no sean None
    tok = processor.tokenizer
    print(f"bos={tok.bos_token_id}, eos={tok.eos_token_id}, pad={tok.pad_token_id}")

    # Configura generation_config explícitamente
    model.generation_config.decoder_start_token_id = tok.bos_token_id or 2
    model.generation_config.bos_token_id           = tok.bos_token_id or 2
    model.generation_config.eos_token_id           = tok.eos_token_id or 2
    model.generation_config.pad_token_id           = tok.pad_token_id or 1
    model.generation_config.max_new_tokens         = 128   # ← usa max_new_tokens, no max_length
    model.generation_config.num_beams              = 4
    model.generation_config.early_stopping         = True

    return processor, model 

def predict(image_path, model, processor):
    img = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=img, return_tensors="pt").pixel_values

    with torch.no_grad():
        ids = model.generate(pixel_values)

    text = processor.batch_decode(ids, skip_special_tokens=True)[0]

    return text.strip()