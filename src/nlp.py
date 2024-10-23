import json
import time
import os
from sentence_transformers import SentenceTransformer, util

# Carregar o modelo Sentence Transformer pré-treinado
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Carregar os dados das cartas (JSON)
with open("./data/merry_cards_data_dump.json", "r", encoding="utf-8") as f:
    cartas = json.load(f)

# Nome do arquivo de cache para embeddings
embeddings_cache_file = "./data/merry_cards_embeddings_cache.json"


# Função para gerar embeddings a partir dos textos relevantes das cartas
def create_card_embedding(carta):
    texts = [
        carta["name"],
        carta["effect"],
        " ".join([crew["name"] for crew in carta["crew"]]),  # Concatenar tripulações
        carta["type"],
        str(carta["power"]),
        str(carta["cost"]),
    ]

    card_text = " ".join([text for text in texts if text])  # Ignora campos vazios
    return model.encode(card_text)


# Função para criar/atualizar o cache de embeddings das cartas
def cache_card_embeddings():
    # Verifica se já existe cache e carrega se disponível
    if os.path.exists(embeddings_cache_file):
        with open(embeddings_cache_file, "r", encoding="utf-8") as f:
            cached_embeddings = json.load(f)
            return cached_embeddings
    else:
        cached_embeddings = {}

    # Verifica se todas as cartas já estão no cache
    for carta in cartas:
        if carta["slug"] not in cached_embeddings:
            card_embedding = create_card_embedding(carta).tolist()  # Converte para lista
            cached_embeddings[carta["slug"]] = card_embedding

    # Salva o cache atualizado
    with open(embeddings_cache_file, "w", encoding="utf-8") as f:
        json.dump(cached_embeddings, f)

    return cached_embeddings


# Função para gerar o embedding a partir dos dados extraídos
def create_extracted_embedding(extracted_data):
    texts = [
        extracted_data["name"],
        extracted_data["description"],
        extracted_data["tribe"],
        extracted_data["type"],
    ]
    extracted_text = " ".join([text for text in texts if text])  # Ignora campos vazios
    return model.encode(extracted_text)


# Função para encontrar a carta mais próxima baseada em embeddings
def find_closest_card(extracted_data, cached_embeddings=None):
    # Gerar o embedding para os dados extraídos
    extracted_embedding = create_extracted_embedding(extracted_data)

    # Carregar os embeddings das cartas do cache
    if cached_embeddings is None:
        with open(embeddings_cache_file, "r", encoding="utf-8") as f:
            cached_embeddings = json.load(f)

    best_match = None
    best_similarity = -1

    # Comparar com os embeddings em cache
    for carta in cartas:
        card_embedding = cached_embeddings[carta["slug"]]
        similarity = util.pytorch_cos_sim(extracted_embedding, card_embedding).item()

        if similarity > best_similarity:
            best_similarity = similarity
            best_match = carta

    return best_match, best_similarity


if __name__ == "__main__":
    # Exemplo de dados extraídos de OCR
    start_time = time.time()
    extracted_data = {
        "power": 8000,
        "cost": 7,
        "description": "On Play / When Attacking DON!! -1 (You may return the specified number of DON!! cards from your field to your DON!! deck.): Up to 1 of your Leader gains +1000 power until the start of your next turn.",
        "name": 'Eustass "Captain" Kid',
        "tribe": "Kid Pirates",
        "type": "CHARACTER",
    }

    # Cachear os embeddings das cartas
    cached_embeddings = cache_card_embeddings()

    # Encontrar a carta mais próxima
    closest_card, similarity = find_closest_card(extracted_data, cached_embeddings)

    print(f"Tempo de execução: {time.time() - start_time:.2f} segundos")
    print(f"Melhor correspondência: {closest_card['name']} ({closest_card['slug']}) com similaridade {similarity}")
    url_path = closest_card["api_url"].split("8000")[1]
    print(f"Api url: http://localhost:8000{url_path}")
    print(f"Ilustrações: {[illustration['external_link'] for illustration in closest_card['illustrations']]}")
