import datetime
import json

import google.generativeai as genai
import os
from dotenv import load_dotenv
import typing_extensions as typing
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from google.generativeai import caching

load_dotenv()

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def create_cache(cache_name):
    with open("data/merry_cards_data_dump.json") as f:
        json_data = json.load(f)

    json_string = json.dumps(json_data)
    print(f"JSON data: {json_string[:100]}...")

    print(f"Creating cache: {cache_name}")
    cache = caching.CachedContent.create(
        model="models/gemini-1.5-flash-001",
        system_instruction=(
            "You have all cards data in cache. "
            "And you are a expert comparing not formatted data with the cards data in cache."
            "And your job is to answer the user's query based on the data in cache."
        ),
        display_name=cache_name,
        contents=[json_string],
        ttl=datetime.timedelta(hours=1),
    )
    print(f"Cache created: {cache}")

    model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    print(f"Model created: {model}")

    return cache.name.split("/")[1]


def extract_text_from_image(image_path):
    card_file = genai.upload_file(image_path)
    model = genai.GenerativeModel("gemini-1.5-flash")

    class ResponseFormat(typing.TypedDict):
        life: int
        attack: int
        cost: int
        counter: int
        type: str
        name: str
        tribe: str
        description: str
        trigger: str

    print(f"Extracting text from image: {image_path}")
    response = model.generate_content(
        [
            "You are a expert extracting text from images.",
            "Extract the maximum text possible from this image.",
            "Do not generate random values, just extract the text from the image.",
            "Send me JSON format",
            card_file,
        ],
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=ResponseFormat
        ),
    )

    print(f"Card extracted: {response.text}")

    return response.text


def get_card_from_ia(cache_name, data_fragment):
    print(f"Retrieving cache: {cache_name}")
    cache = caching.CachedContent.get(name=cache_name)

    print(f"Cache retrieved: {cache.name}")

    model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    print(f"Model created: {model}")

    class Card(typing.TypedDict):
        api_url: str
        confidence: float

    class ResponseFormat(typing.TypedDict):
        possible_cards: typing.List[Card]

    print(f"Extracting card from cache: {cache_name}")
    response = model.generate_content(
        [
            "I extract these data fragments from an image, based on the cached data,"
            "which cards are closest to this data? "
            "Return a list of possible cards with their confidence, sorted by confidence.",
            "Max 5 possible cards",
            f"Data fragments: {data_fragment}",
        ],
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        },
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json", response_schema=ResponseFormat
        ),
    )

    print(f"Card found: {response.text}")

    return response.text


if __name__ == "__main__":
    # just call once to create the cache, because the api has a token limit per hour
    # model_cache_name = create_cache("merry-cards-cache")
    model_cache_name = "limi2u15lp0c"

    data = extract_text_from_image("../images/20241020_211624.jpg")
    result = get_card_from_ia(cache_name=model_cache_name, data_fragment=data)

    with open("data/card_recognition_output.json", "w") as file:
        file.write(
            json.dumps(
                {
                    "extracted_data": json.loads(data),
                    "search_result": json.loads(result),
                },
                indent=4,
            )
        )

    # Terminal output:
    """
        Extracting text from image: ../images/20241020_211624.jpg
        Card extracted: {"attack": 2000, "counter": 1000, "description": "On Play Look at 5 cards from the top of your deck; reveal up to 1 {Straw Hat Crew} type card other than [Nami] and add it to your hand. Then, place the rest at the bottom of your deck in any order.", "name": "Nami", "tribe": "Straw Hat Crew", "type": "CHARACTER"}
        
        Retrieving cache: limi2u15lp0c
        Cache retrieved: cachedContents/limi2u15lp0c
        Model created: genai.GenerativeModel(
            model_name='models/gemini-1.5-flash-001',
            generation_config={},
            safety_settings={},
            tools=None,
            system_instruction=None,
            cached_content=cachedContents/limi2u15lp0c
        )
        Extracting card from cache: limi2u15lp0c
        Card found: {"possible_cards": [{"api_url": "http://172.24.160.1:8000/cards/OP01-016/", "confidence": 1.0}, {"api_url": "http://172.24.160.1:8000/cards/OP03-030/", "confidence": 1.0}, {"api_url": "http://172.24.160.1:8000/cards/OP05-064/", "confidence": 1.0}, {"api_url": "http://172.24.160.1:8000/cards/OP06-025/", "confidence": 1.0}, {"api_url": "http://172.24.160.1:8000/cards/OP07-022/", "confidence": 1.0}]}
    """
