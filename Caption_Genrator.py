# Image_caption_generator
import tensorflow as tf
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import requests

# Load the pre-trained model and processor
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

def generate_caption(image_path):
    # Load and process the image
    image = Image.open(image_path)
    pixel_values = image_processor(images=image, return_tensors="pt").pixel_values

    # Generate caption
    output_ids = model.generate(pixel_values, max_length=50, num_beams=4, early_stopping=True)
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return caption

# Example usage
if __name__ == "__main__":  
    # Replace 'your_image.jpg' with the path to your image file
    image_path = '/content/R.jpg'
    caption = generate_caption(image_path)
    print("Generated Caption:", caption)
