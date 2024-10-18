import json

from PIL import Image
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

media = Image.open("../images/OP07-109_p2.png")

myfile = genai.upload_file("../images/OP07-109_p2.png")

model = genai.GenerativeModel("gemini-1.5-flash")

json_extructed_text_example = {
    "life": 0,
    "attack": 0,
    "cost": 0,
    "counter": 0,
    "type": "Leader",
    "name": "Name",
    "tribe": "Land..../Cla...",
    "description": "When this card enters the field, ",
    "trigger": "",
}

result = model.generate_content(
    [
        myfile,
        "\n\n",
        "Extract the text from this image and generate a json with the following "
        f"structure:\n {json_extructed_text_example}",
        "Send the a string with the json generated only, without any other characters "
        "so that I can convert it to json without any problems.",
        "Do not include especial characters in the json like break lines.",
        "The json needs to use a double quote for the keys and values.",
    ],
)

try:
    parsed_json = json.dumps(json.loads(result.text), indent=4)
    with open("from_image.json", "w") as f:
        f.write(parsed_json)
except json.JSONDecodeError as e:
    print(result.text)
    print(f"Erro ao decodificar JSON: {e}")
