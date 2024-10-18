import json

import pytesseract
from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente do arquivo .env
load_dotenv()
# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def extract_text_from_image(image_path):
    # Open an image file
    with Image.open(image_path) as img:
        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(img)
    return text


def send_text_to_gemini(extracted_text):
    json_extructed_text_example = {
        "cost": int,
        "type": "Leader",
        "name": "Name",
        "tribe": "Land..../Cla...",
        "life": int,
        "attack": int,
        "description": "When this card enters the field, "
        "you may put a card from your mana zone into the battle zone.",
    }

    model = genai.GenerativeModel("gemini-1.5-flash")
    # Prompt to extructore a json from extracted_text based on the json_extructed_text_example
    prompt = (
        f"From this extracted text\n{extracted_text}\n"
        f"Generate a json with the following structure:\n{json_extructed_text_example}\n"
        f"Send the a string with the json generated only, without any other characters."
    )

    return model.generate_content(prompt)


if __name__ == "__main__":
    image_path = "../images/EB01-001.png"
    extracted_text = extract_text_from_image(image_path)

    response = send_text_to_gemini(extracted_text).text
    # response is a json string, print it as a json object in the console indenting it
    print(json.dumps(json.loads(response, strict=False), indent=4))
