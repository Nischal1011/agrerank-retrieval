# Injects noise chunks into the corpus for stress-testing retrieval methods

import argparse
import json
import numpy as np
from pathlib import Path
import sys
import os

# For generating noise embeddings
try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

MODEL_EMBEDDING = "text-embedding-3-large"

def l2n(V: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    if V.ndim == 1:
        n = np.linalg.norm(V)
        return V / max(n, eps)
    n = np.linalg.norm(V, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return V / n

class OAI:
    def __init__(self, bs: int = 128):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("Error: OPENAI_API_KEY not set.")
        self.c = OpenAI(api_key=api_key)
        self.bs = bs

    def encode(self, texts: list) -> np.ndarray:
        if not texts:
            return np.zeros((0, 3072), dtype=np.float32)
        out = []
        for i in range(0, len(texts), self.bs):
            batch = texts[i:i + self.bs]
            batch = [t if t.strip() else " " for t in batch]
            try:
                resp = self.c.embeddings.create(model=MODEL_EMBEDDING, input=batch)
                out.append(np.array([d.embedding for d in resp.data], dtype=np.float32))
            except Exception as e:
                print(f"Embedding error: {e}", file=sys.stderr)
                out.append(np.zeros((len(batch), 3072), dtype=np.float32))
        return l2n(np.vstack(out)).astype(np.float32)


# Noise text templates - diverse domains to create realistic distractors
NOISE_TEMPLATES = [
    # Sports
    "The {team} defeated the {opponent} with a final score of {score}. The game was held at {stadium} in front of {crowd} fans.",
    "In the {year} {sport} championship, {player} scored {points} points leading the team to victory.",
    "{player} signed a {years}-year contract worth ${amount} million with the {team}.",
    
    # Entertainment
    "The movie '{title}' starring {actor} grossed ${amount} million at the box office in its opening weekend.",
    "{celebrity} announced their new album '{album}' will be released on {date}.",
    "The TV series '{show}' was renewed for season {season} by {network}.",
    
    # Business/Finance
    "{company} reported quarterly earnings of ${amount} billion, {change}% compared to last year.",
    "The stock price of {ticker} rose {percent}% following the announcement of {event}.",
    "{company} announced plans to acquire {target} for ${amount} billion.",
    
    # Weather
    "The weather forecast for {city} shows {condition} with temperatures reaching {temp} degrees.",
    "A {severity} weather warning has been issued for {region} due to {weather_event}.",
    
    # Food/Recipes
    "To make {dish}, combine {ingredient1}, {ingredient2}, and {ingredient3}. Bake at {temp} degrees for {time} minutes.",
    "The restaurant {name} in {city} was awarded {stars} Michelin stars for their {cuisine} cuisine.",
    
    # Travel
    "Flights from {origin} to {destination} start at ${price} during the {season} season.",
    "The {hotel} hotel in {city} offers {amenity} and has a rating of {rating} stars.",
    
    # Technology (non-ML)
    "The new {brand} {device} features a {size}-inch display and {battery} mAh battery.",
    "{company} released version {version} of their {software} software with new features including {feature}.",
    
    # Health/Fitness
    "A study found that {activity} for {duration} minutes per day can reduce {condition} by {percent}%.",
    "The FDA approved {drug} for treatment of {condition} in patients over {age} years old.",
    
    # History
    "In {year}, {historical_figure} {achievement} which led to {consequence}.",
    "The {event} of {year} marked a turning point in {field} history.",
    
    # Geography
    "The {landmark} in {country} attracts over {visitors} million visitors annually.",
    "{city} is the capital of {country} with a population of approximately {population} million.",
]

# Fill-in values for templates
FILL_VALUES = {
    "team": ["Lakers", "Patriots", "Yankees", "Manchester United", "Real Madrid", "Warriors", "Cowboys"],
    "opponent": ["Celtics", "Giants", "Red Sox", "Liverpool", "Barcelona", "Cavaliers", "Eagles"],
    "score": ["98-95", "24-21", "5-3", "2-1", "3-0", "110-105", "31-28"],
    "stadium": ["Madison Square Garden", "Wembley Stadium", "Camp Nou", "Yankee Stadium", "AT&T Stadium"],
    "crowd": ["18,000", "75,000", "90,000", "45,000", "80,000"],
    "year": ["2019", "2020", "2021", "2022", "2023", "2024"],
    "sport": ["basketball", "football", "soccer", "baseball", "hockey", "tennis"],
    "player": ["James", "Messi", "Brady", "Curry", "Ronaldo", "Mahomes", "Ohtani"],
    "points": ["32", "45", "28", "41", "38", "25", "3"],
    "years": ["3", "4", "5", "6", "7", "10"],
    "amount": ["50", "100", "200", "350", "500", "1.2", "2.5", "4.8"],
    "title": ["The Last Journey", "Midnight Sun", "Ocean's Rise", "Silent Echo", "Dark Horizon"],
    "actor": ["Smith", "Johnson", "Williams", "Davis", "Martinez", "Anderson"],
    "celebrity": ["Taylor Swift", "Drake", "Beyoncé", "Ed Sheeran", "The Weeknd"],
    "album": ["Midnight Dreams", "Golden Hour", "Electric Soul", "Velvet Sky"],
    "date": ["March 15", "June 1", "September 22", "November 8", "January 30"],
    "show": ["Breaking Dawn", "The Crown", "Stranger Things", "The Office", "Game of Thrones"],
    "season": ["4", "5", "6", "7", "8"],
    "network": ["Netflix", "HBO", "Amazon Prime", "Disney+", "Apple TV+"],
    "company": ["Apple", "Google", "Microsoft", "Amazon", "Meta", "Tesla", "Nvidia"],
    "ticker": ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NVDA"],
    "change": ["+15", "+8", "-3", "+22", "-7", "+45"],
    "percent": ["5.2", "3.8", "12.4", "7.1", "2.9", "18.3"],
    "event": ["the merger", "new product launch", "earnings beat", "CEO appointment"],
    "target": ["Startup Inc", "Tech Corp", "Digital Solutions", "Cloud Systems"],
    "city": ["New York", "London", "Tokyo", "Paris", "Sydney", "Dubai", "Singapore"],
    "condition": ["sunny", "cloudy", "rainy", "snowy", "partly cloudy", "thunderstorms"],
    "temp": ["72", "85", "45", "32", "68", "91"],
    "severity": ["severe", "moderate", "extreme", "winter storm", "heat"],
    "region": ["the Northeast", "Southern California", "the Midwest", "Florida", "Texas"],
    "weather_event": ["heavy snow", "flooding", "high winds", "extreme heat", "ice storms"],
    "dish": ["pasta carbonara", "chicken tikka", "beef stew", "vegetable stir-fry", "chocolate cake"],
    "ingredient1": ["flour", "chicken", "beef", "tomatoes", "onions"],
    "ingredient2": ["eggs", "garlic", "olive oil", "butter", "cream"],
    "ingredient3": ["cheese", "herbs", "spices", "vegetables", "wine"],
    "time": ["30", "45", "60", "90", "120"],
    "name": ["Le Bernardin", "Noma", "Eleven Madison Park", "The French Laundry"],
    "stars": ["1", "2", "3"],
    "cuisine": ["French", "Italian", "Japanese", "American", "Nordic"],
    "origin": ["New York", "Los Angeles", "Chicago", "Miami", "Boston"],
    "destination": ["London", "Paris", "Tokyo", "Rome", "Barcelona"],
    "price": ["299", "450", "599", "899", "1200"],
    "hotel": ["Ritz-Carlton", "Four Seasons", "Marriott", "Hilton", "Hyatt"],
    "amenity": ["spa services", "rooftop pool", "private beach", "golf course"],
    "rating": ["4", "4.5", "5"],
    "brand": ["Samsung", "Apple", "Sony", "LG", "Google"],
    "device": ["smartphone", "tablet", "laptop", "smartwatch", "TV"],
    "size": ["6.1", "6.7", "10.9", "13.3", "15.6", "55", "65"],
    "battery": ["3500", "4500", "5000", "6000"],
    "version": ["2.0", "3.5", "4.0", "5.1", "10.2"],
    "software": ["productivity", "design", "database", "security"],
    "feature": ["dark mode", "AI assistant", "cloud sync", "collaboration tools"],
    "activity": ["walking", "swimming", "cycling", "yoga", "running"],
    "duration": ["30", "45", "60", "20"],
    "drug": ["Nexavar", "Keytruda", "Humira", "Ozempic"],
    "age": ["18", "21", "40", "50", "65"],
    "historical_figure": ["Napoleon", "Churchill", "Lincoln", "Einstein", "Curie"],
    "achievement": ["discovered penicillin", "signed the treaty", "invented the telephone", "led the revolution"],
    "consequence": ["major changes in medicine", "peace in Europe", "the industrial age", "independence"],
    "field": ["scientific", "political", "economic", "cultural"],
    "landmark": ["Eiffel Tower", "Great Wall", "Colosseum", "Taj Mahal", "Machu Picchu"],
    "country": ["France", "China", "Italy", "India", "Peru", "Japan", "Brazil"],
    "visitors": ["7", "10", "15", "6", "2.5"],
    "population": ["8.3", "14", "9.5", "21", "12.5"],
}

def generate_noise_text(n: int, seed: int = 42) -> list:
    """Generate n noise text chunks."""
    np.random.seed(seed)
    texts = []
    
    for i in range(n):
        template = np.random.choice(NOISE_TEMPLATES)
        text = template
        
        # Fill in placeholders
        import re
        placeholders = re.findall(r'\{(\w+)\}', template)
        for ph in placeholders:
            if ph in FILL_VALUES:
                value = np.random.choice(FILL_VALUES[ph])
                text = text.replace('{' + ph + '}', value, 1)
        
        texts.append(text)
    
    return texts


def main():
    parser = argparse.ArgumentParser(description="Create noisy corpus for stress testing")
    parser.add_argument("--chunks", type=Path, required=True, help="Original corpus JSONL")
    parser.add_argument("--embeddings", type=Path, required=True, help="Original embeddings .npy")
    parser.add_argument("--noise_count", type=int, default=10000, help="Number of noise chunks to add")
    parser.add_argument("--out_chunks", type=Path, required=True, help="Output noisy corpus JSONL")
    parser.add_argument("--out_embeddings", type=Path, required=True, help="Output noisy embeddings .npy")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    print("=" * 70)
    print("Creating Noisy Corpus")
    print("=" * 70)
    
    # Load original corpus
    print(f"Loading original corpus from {args.chunks}...")
    original_chunks = []
    with args.chunks.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                original_chunks.append(json.loads(line))
    print(f"  Loaded {len(original_chunks)} original chunks")
    
    # Load original embeddings
    print(f"Loading original embeddings from {args.embeddings}...")
    original_emb = np.load(args.embeddings).astype(np.float32)
    print(f"  Shape: {original_emb.shape}")
    
    # Generate noise text
    print(f"\nGenerating {args.noise_count} noise chunks...")
    noise_texts = generate_noise_text(args.noise_count, seed=args.seed)
    
    # Create noise chunk records
    noise_chunks = []
    for i, text in enumerate(noise_texts):
        noise_chunks.append({
            "chunk_id": f"noise_{i:06d}",
            "text": text,
            "is_noise": True
        })
    
    # Embed noise chunks
    print(f"Embedding noise chunks (this may take a while)...")
    encoder = OAI()
    
    batch_size = 500
    noise_embeddings = []
    for i in range(0, len(noise_texts), batch_size):
        batch = noise_texts[i:i+batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(len(noise_texts) + batch_size - 1)//batch_size}...")
        emb = encoder.encode(batch)
        noise_embeddings.append(emb)
    
    noise_emb = np.vstack(noise_embeddings)
    print(f"  Noise embeddings shape: {noise_emb.shape}")
    
    # Combine original + noise
    print("\nCombining original and noise...")
    combined_chunks = original_chunks + noise_chunks
    combined_emb = np.vstack([original_emb, noise_emb])
    
    print(f"  Combined chunks: {len(combined_chunks)}")
    print(f"  Combined embeddings: {combined_emb.shape}")
    
    # Save
    print(f"\nSaving noisy corpus to {args.out_chunks}...")
    with args.out_chunks.open("w", encoding="utf-8") as f:
        for chunk in combined_chunks:
            f.write(json.dumps(chunk) + "\n")
    
    print(f"Saving noisy embeddings to {args.out_embeddings}...")
    np.save(args.out_embeddings, combined_emb)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Original chunks: {len(original_chunks)}")
    print(f"Noise chunks: {len(noise_chunks)}")
    print(f"Total chunks: {len(combined_chunks)}")
    print(f"Noise ratio: {len(noise_chunks) / len(combined_chunks) * 100:.1f}%")


if __name__ == "__main__":
    main()

# Usage:
# python vectorrag/noisy_corpus.py\
#     --chunks data/master_in_embedding_order.jsonl \
#     --embeddings data/embeddings/fixed250/corpus_emb.npy \
#     --noise_count 10000 \
#     --out_chunks data/noisy_corpus_10k.jsonl \
#     --out_embeddings data/noisy_embeddings_10k.npy