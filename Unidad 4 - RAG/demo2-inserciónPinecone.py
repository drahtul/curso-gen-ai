from sentence_transformers import SentenceTransformer  # type: ignore
import os
from dotenv import load_dotenv
from huggingface_hub import login
from pinecone import Pinecone  # type: ignore

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "obli-rag-cine"

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

movies = [
    {"id": "movie-01", "title": "The Matrix", "text": "A hacker discovers reality is a simulation and joins a rebellion to free humanity from intelligent machines.", "rating": 8.7, "year": 1999},
    {"id": "movie-02", "title": "Inception", "text": "A skilled thief enters dreams to steal secrets and plants an idea in a target's mind through layered dream worlds.", "rating": 8.8, "year": 2010},
    {"id": "movie-03", "title": "Toy Story", "text": "A cowboy doll fears replacement by a spaceman action figure and learns to share affection and friendship.", "rating": 8.3, "year": 1995},
    {"id": "movie-04", "title": "The Hangover", "text": "Three friends wake up after a chaotic bachelor party in Las Vegas with no memory of the previous night.", "rating": 7.7, "year": 2009},
    {"id": "movie-05", "title": "La La Land", "text": "A jazz pianist and an aspiring actress fall in love in Los Angeles while chasing their dreams.", "rating": 8.0, "year": 2016},
    {"id": "movie-06", "title": "Avatar", "text": "A paraplegic marine joins a mission to Pandora and becomes torn between duty and his connection to the planet's native people.", "rating": 7.8, "year": 2009},
    {"id": "movie-07", "title": "The Lion King", "text": "A young lion cub learns courage and responsibility after tragedy forces him to confront his destiny.", "rating": 8.5, "year": 1994},
    {"id": "movie-08", "title": "Finding Nemo", "text": "A clownfish searches the ocean for his son after he is captured and taken away from the reef.", "rating": 7.8, "year": 2003},
    {"id": "movie-09", "title": "The Godfather", "text": "The aging patriarch of a crime family tries to keep his empire together while his son enters the violent world of organized crime.", "rating": 9.2, "year": 1972},
    {"id": "movie-10", "title": "Pulp Fiction", "text": "Intertwined stories of crime, redemption, and violence unfold through a nonlinear narrative in Los Angeles.", "rating": 8.9, "year": 1994},
    {"id": "movie-11", "title": "Back to the Future", "text": "A teenage boy travels to the past in a time machine and accidentally alters the lives of his family.", "rating": 8.5, "year": 1985},
    {"id": "movie-12", "title": "The Dark Knight", "text": "Batman faces the Joker in a battle that tests the limits of justice, chaos, and morality in Gotham.", "rating": 9.0, "year": 2008},
    {"id": "movie-13", "title": "Forrest Gump", "text": "A kind and simple man witnesses decades of history while living an extraordinary life full of love and perseverance.", "rating": 8.8, "year": 1994},
    {"id": "movie-14", "title": "Gravity", "text": "Astronauts stranded in orbit fight to survive after a debris storm destroys their mission and threatens their lives.", "rating": 7.7, "year": 2013},
    {"id": "movie-15", "title": "Interstellar", "text": "A team of astronauts travels through a wormhole to save humanity from a dying Earth and seek a new home.", "rating": 8.7, "year": 2014},
    {"id": "movie-16", "title": "Casablanca", "text": "A former nightclub owner and a resistance fighter confront love, loyalty, and wartime choices in Morocco.", "rating": 8.5, "year": 1942},
    {"id": "movie-17", "title": "The Social Network", "text": "A brilliant programmer creates a social media platform and becomes entangled in lawsuits, betrayal, and ambition.", "rating": 7.8, "year": 2010},
    {"id": "movie-18", "title": "Inside Out", "text": "The emotions inside a young girl's mind guide her through moving to a new city and coping with change.", "rating": 8.1, "year": 2015},
    {"id": "movie-19", "title": "The Internship", "text": "Two middle-aged job seekers join a technology internship and discover the value of adaptability and teamwork.", "rating": 6.3, "year": 2013},
    {"id": "movie-20", "title": "Superbad", "text": "Two high school friends try to secure alcohol for a party and end up in a series of wildly awkward misadventures.", "rating": 7.6, "year": 2007},
    {"id": "movie-21", "title": "Crazy Rich Asians", "text": "A Chinese-American professor travels to Singapore and confronts family expectations, romance, and cultural identity.", "rating": 7.0, "year": 2018},
    {"id": "movie-22", "title": "The Proposal", "text": "A demanding editor forces her assistant to marry her so she can keep her visa, leading to a comedy of romance and chaos.", "rating": 6.7, "year": 2009},
    {"id": "movie-23", "title": "Bridesmaids", "text": "A maid of honor navigates wedding drama, friendship, and embarrassing mishaps while trying to support her best friend.", "rating": 6.8, "year": 2011},
    {"id": "movie-24", "title": "Legally Blonde", "text": "A bubbly sorority girl uses her intelligence and determination to earn a law degree and prove herself in a male-dominated world.", "rating": 6.4, "year": 2001},
    {"id": "movie-25", "title": "Mean Girls", "text": "A teenager navigates social hierarchy and gossip at a new school while trying to fit in without losing herself.", "rating": 7.1, "year": 2004},
    {"id": "movie-26", "title": "Zoolander", "text": "A famous male model becomes entangled in a ridiculous conspiracy while trying to remain relevant in the fashion world.", "rating": 6.5, "year": 2001},
    {"id": "movie-27", "title": "Groundhog Day", "text": "A cynical weatherman relives the same day repeatedly and learns about growth, love, and self-improvement.", "rating": 8.1, "year": 1993},
    {"id": "movie-28", "title": "The Devil Wears Prada", "text": "A young aspiring journalist takes a demanding job with a famous fashion editor and learns about ambition and sacrifice.", "rating": 6.9, "year": 2006},
    {"id": "movie-29", "title": "Ferris Bueller's Day Off", "text": "A clever teenager skips school for one last great adventure in Chicago while evading his principal.", "rating": 7.8, "year": 1986},
    {"id": "movie-30", "title": "The Grand Budapest Hotel", "text": "A concierge and his protégé become entangled in a dramatic theft involving a priceless painting and a mysterious family.", "rating": 8.1, "year": 2014},
    {"id": "movie-31", "title": "Shrek", "text": "A grumpy ogre is sent on a quest to rescue a princess and discover that true love is about more than appearances.", "rating": 7.9, "year": 2001},
    {"id": "movie-32", "title": "Kung Fu Panda", "text": "A clumsy panda dreams of becoming a kung fu master and learns that training and heart matter more than talent.", "rating": 7.8, "year": 2008},
    {"id": "movie-33", "title": "The Intouchables", "text": "A wealthy quadriplegic hires a young man from the projects as his caregiver, creating a bond based on humor and humanity.", "rating": 8.5, "year": 2011},
    {"id": "movie-34", "title": "Paddington", "text": "A kind-hearted bear from Peru arrives in London and wins over a family with warmth, curiosity, and humor.", "rating": 7.4, "year": 2014},
    {"id": "movie-35", "title": "Chef", "text": "A talented chef quits his restaurant and launches a food truck to reconnect with his family and rediscover his passion.", "rating": 7.3, "year": 2014},
    {"id": "movie-36", "title": "The Internship", "text": "Two former salesmen compete for a chance to work at Google and learn about innovation, teamwork, and reinvention.", "rating": 6.3, "year": 2013},
    {"id": "movie-37", "title": "School of Rock", "text": "A failed musician poses as a substitute teacher and turns a class of children into a rock band with unexpected results.", "rating": 7.1, "year": 2003},
    {"id": "movie-38", "title": "Dumb and Dumber", "text": "Two dim-witted friends travel across the country to return a suitcase and get caught in a chaotic adventure.", "rating": 6.3, "year": 1994},
    {"id": "movie-39", "title": "The Princess Diaries", "text": "A shy teenager discovers she is heir to a European throne and embraces a new identity with comedy and heart.", "rating": 6.3, "year": 2001},
    {"id": "movie-40", "title": "Hot Fuzz", "text": "A super-efficient London cop is reassigned to a sleepy village where he uncovers a violent conspiracy hidden beneath the town's charm.", "rating": 7.8, "year": 2007},
    {"id": "movie-41", "title": "The Nice Guys", "text": "A private investigator and an enforcer investigate a missing woman and uncover a web of corruption and murder.", "rating": 7.4, "year": 2016},
    {"id": "movie-42", "title": "The Hangover Part II", "text": "The gang heads to Thailand for another wedding and wakes up in a bizarre mess with missing memories and chaos.", "rating": 6.5, "year": 2011},
    {"id": "movie-43", "title": "Mamma Mia!", "text": "A young woman secretly invites three men to her wedding in hopes of discovering who her father is, leading to musical chaos.", "rating": 6.4, "year": 2008},
    {"id": "movie-44", "title": "About Time", "text": "A young man discovers he can travel through time and uses the gift to fix regrets while learning the value of love.", "rating": 7.8, "year": 2013},
    {"id": "movie-45", "title": "Moulin Rouge!", "text": "A writer falls in love with a courtesan in Paris while their lives are transformed by romance, passion, and drama.", "rating": 7.6, "year": 2001},
    {"id": "movie-46", "title": "The Bucket List", "text": "Two terminally ill men escape a hospital and set out to complete a list of personal dreams before time runs out.", "rating": 7.4, "year": 2007},
    {"id": "movie-47", "title": "Absolutely Fabulous", "text": "Two fashion-obsessed friends navigate life, relationships, and absurd situations with outrageous humor and style.", "rating": 6.7, "year": 1992},
    {"id": "movie-48", "title": "Napoleon Dynamite", "text": "A quirky teenager and his family navigate small-town life with awkward humor, awkward romance, and unforgettable characters.", "rating": 6.9, "year": 2004},
    {"id": "movie-49", "title": "The Best Exotic Marigold Hotel", "text": "A group of senior citizens move to India and discover new friendships, purpose, and a richer sense of adventure.", "rating": 7.2, "year": 2011},
    {"id": "movie-50", "title": "The Grand Seduction", "text": "A small Canadian town tries to attract a doctor by staging a convincing community experience, leading to comic cultural chaos.", "rating": 6.7, "year": 2013},
]

vectors = []
for movie in movies:
    embedding = model.encode(movie["text"]).tolist()
    vectors.append({
        "id": movie["id"],
        "values": embedding,
        "metadata": {
            "title": movie["title"],
            "text": movie["text"],
            "rating": float(movie["rating"]),
            "year": str(movie["year"])
        }
    })

# insertar en batches
for i in range(0, len(vectors), 20):
    batch = vectors[i:i + 20]
    index.upsert(vectors=batch)
    print(f"✅ Insertados {i + 1} a {min(i + 20, len(vectors))} registros")

print("✅ 50 películas insertadas en Pinecone")
print(index.describe_index_stats())