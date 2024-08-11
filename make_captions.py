import os
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

# Initialize model and processor
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)

# Define data path
data_path = "yourpath"

# Define prompt for caption generation
prompt = "<MORE_DETAILED_CAPTION>"

# Function to resize image while maintaining aspect ratio
def resize_image(image, max_size=1024):
    width, height = image.size
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int(max_size * height / width)
        else:
            new_height = max_size
            new_width = int(max_size * width / height)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# Function to process each image and save the caption
def generate_caption_for_image(image_path, folder_basename):
    try:
        image = Image.open(image_path)

        # Handle PNGs by converting them to RGB mode
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        
        # Resize image
        image = resize_image(image)
        
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False
        )
        
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        
        # Remove special tokens
        cleaned_text = generated_text.replace("<s>", "").replace("</s>", "").strip()
        
        # Add folder basename to the beginning of the caption
        final_caption = f"{folder_basename}, {cleaned_text}"
        
        return final_caption
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        os.remove(image_path)
        print(f"Deleted {image_path} due to the error.")
        return None

# Get the basename of the folder
folder_basename = os.path.basename(os.path.normpath(data_path))

# Process all images in the folder
for filename in os.listdir(data_path):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        image_path = os.path.join(data_path, filename)
        final_caption = generate_caption_for_image(image_path, folder_basename)
        
        if final_caption:
            # Print the output of the Florence model
            print(f"Generated caption for {filename}:")
            print(final_caption)
            
            # Save caption to .txt file
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(data_path, txt_filename)
            
            with open(txt_filepath, "w") as txt_file:
                txt_file.write(final_caption)

            print(f"Caption saved for {filename} as {txt_filename}\n")
        else:
            print(f"Skipping {filename} due to error.")