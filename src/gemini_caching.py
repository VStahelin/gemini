import json
import datetime
import google.generativeai as genai
from google.generativeai import caching

import os

genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def create_cache(cache_name: str = "merry-cards-cache"):
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

    response = model.generate_content(
        [
            (
                "Based on the data in cache, how many with 'rare' equal to 'SR'?, return all 'api_url' of "
                "the cards with 'rare' equal to 'SR'."
            )
        ]
    )

    print(response.usage_metadata)
    print(response.text)


def test_cache(cache_name: str = "merry-cards-cache"):
    print(f"Retrieving cache: {cache_name}")
    cache = caching.CachedContent.get(name=cache_name)
    print(f"Cache retrieved: {cache.name}")

    model = genai.GenerativeModel.from_cached_content(cached_content=cache)
    print(f"Model created: {model}")

    response = model.generate_content(
        [
            (
                "I extract these data fragments from an image, based on the cached data, which cards are closest to this data?"
                "Data fragments: 'rare': 'SR', 'type': 'Leader', 'life': 5000, 'attack': 5000"
            )
        ]
    )

    print(response.usage_metadata)
    print(response.text)


if __name__ == "__main__":
    # create_cache(cache_name="merry-cards-cache")
    test_cache(cache_name="bx8i0b93m6sh")
