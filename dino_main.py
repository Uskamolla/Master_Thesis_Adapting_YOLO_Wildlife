from groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import os
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the model
model = load_model("groundingdino/config/GroundingDINO_SwinT_OGC.py", "weights/groundingdino_swint_ogc.pth")

# Directory containing images
IMAGE_DIR = "/osm/uskamolla/Master_Thesis/DINO4/report_results/rabbit/raccoon/images"

# Parameters
TEXT_PROMPT = "raccoon."
BOX_TRESHOLD = 0.3
TEXT_TRESHOLD = 0.3

output_dir = "/osm/uskamolla/Master_Thesis/DINO4/report_results/rabbit/raccoon/output5"
label_dir = os.path.join(output_dir, "label")

# Create label directory if it doesn't exist
if not os.path.exists(label_dir):
    os.makedirs(label_dir)

# Save the parameters to a .txt file
parameters_path = os.path.join(output_dir, "parameters.txt")
with open(parameters_path, 'w') as param_file:
    param_file.write(f"TEXT_PROMPT = {TEXT_PROMPT}\n")
    param_file.write(f"BOX_TRESHOLD = {BOX_TRESHOLD}\n")
    param_file.write(f"TEXT_TRESHOLD = {TEXT_TRESHOLD}\n")

# Iterate over files in the image directory
for filename in os.listdir(IMAGE_DIR):
    if filename.endswith((".png", ".jpg", ".jpeg")):  # Add other file types if needed
        image_path = os.path.join(IMAGE_DIR, filename)
        image_source, image = load_image(image_path)

        boxes, logits, phrases = predict(
            model=model,
            image=image,
            caption=TEXT_PROMPT,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

        # Save the annotated image
        annotated_image_path = os.path.join(output_dir, "annotated_" + filename)
        cv2.imwrite(annotated_image_path, annotated_frame)

        # Save the coordinates to a .txt file in the label folder
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_path = os.path.join(label_dir, txt_filename)
        with open(txt_path, 'w') as file:
            for box in boxes:
                # Write the original box coordinates to the file
                file.write(f"{box[0]} {box[1]} {box[2]} {box[3]}\n")
