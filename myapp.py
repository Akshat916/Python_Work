import click
import csv
import time
from collections import Counter
import pymongo
import nltk
import spacy
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load the spaCy English language model
nlp = spacy.load('en_core_web_sm')

# Download the VADER sentiment analysis model (run this once)
nltk.download('vader_lexicon')

# Initialize the VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()

csv_data = []

myClient = pymongo.MongoClient("mongodb://localhost:27017/")
db = myClient['test']
collection = db['news_headline']

@click.group()
def myapp():
    pass

@click.command()
@click.argument('csv_file', type=click.Path(exists=True))
def import_headlines(csv_file):
    start_time = time.time()
    # Read the CSV file and store the data
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        # Skip the header row if present
        next(reader, None)
        for row in reader:
            csv_data.append({
                'headlineId': row[0],
                'headline': row[1]
            })
    # Insert data into MongoDB
    collection.insert_many(csv_data)
    end_time = time.time()
    execution_time = end_time - start_time
    click.echo(f"Data ingestion completed in {execution_time:.2f} seconds.")

@click.command()
def extract_entities_and_analyze():
    start_time = time.time()

    # Process headlines from MongoDB
    entities_counter = Counter()

    for doc in collection.find({}):
        id = doc['_id']
        text = doc['headline']
        doc = nlp(text)

        entities_counter.update([(ent.text, ent.label_) for ent in doc.ents])

        sentiment_info = perform_sentiment_analysis_nltk(text)

        condition = {"_id": id}
        for ent in doc.ents:
            # Define the update operation
            update_operation = {
                "$set": {
                    "entities": {
                        "entity": {
                            "text": ent.text,
                            "type": ent.label_
                        },
                    },
                    "sentimentAnalysis": sentiment_info['sentiment_label'],
                }
            }
        collection.update_many(condition, update_operation)

    end_time = time.time()
    execution_time = end_time - start_time
    click.echo(f"Entity extraction and sentiment analysis completed in {execution_time:.2f} seconds.")

@click.command()
def top100entitieswithtype():
    start_time = time.time()

    # Initialize a counter for entities
    entities_counter = Counter()

    # Query the MongoDB collection to aggregate entity frequencies and types
    pipeline = [
        {
            "$unwind": "$entities"  # Unwind the entities array
        },
        {
            "$match": {
                "entities.entity.text": {"$exists": True, "$ne": None},  # Exclude null or non-existent values
                "entities.entity.type": {"$exists": True, "$ne": None}
            }
        },
        {
            "$group": {
                "_id": {
                    "text": "$entities.entity.text",
                    "type": "$entities.entity.type"
                },
                "count": {"$sum": 1}
            }
        },
        {
            "$sort": {"count": -1}  # Sort by count in descending order
        },
        {
            "$limit": 100  # Limit to the top 100 entities
        }
    ]

    result = collection.aggregate(pipeline)

    for entry in result:
        entity_text = entry["_id"]["text"]
        entity_type = entry["_id"]["type"]
        frequency = entry["count"]
        print(f"Entity: {entity_text}, Type: {entity_type}, Frequency: {frequency}")

    end_time = time.time()
    execution_time = end_time - start_time
    click.echo(f"Top 100 entities retrieval completed in {execution_time:.2f} seconds.")

@click.command()
@click.argument('entity_name')
def allheadlinesfor(entity_name):
    start_time = time.time()

    # Query the MongoDB collection to find all headlines for the specified entity name
    pipeline = [
        {
            "$unwind": "$entities"  # Unwind the entities array
        },
        {
            "$match": {
                "entities.entity.text": entity_name  # Match documents with the specified entity name
            }
        },
        {
            "$project": {
                "_id": 0,
                "headline": 1
            }
        }
    ]

    result = collection.aggregate(pipeline)

    headlines = [entry["headline"] for entry in result if entry.get("headline")]

    end_time = time.time()
    execution_time = end_time - start_time

    if headlines:
        for i, headline in enumerate(headlines, start=1):
            print(f"{i}. {headline}")
    else:
        print(f"No headlines found for entity: {entity_name}")

    print(f"Retrieval completed in {execution_time:.2f} seconds.")


def perform_sentiment_analysis_nltk(text):
    sentiment_scores = sia.polarity_scores(text)

    compound_score = sentiment_scores['compound']

    if compound_score >= 0.05:
        sentiment_label = "Positive"
    elif compound_score <= -0.05:
        sentiment_label = "Negative"
    else:
        sentiment_label = "Neutral"

    return {
        'compound_score': compound_score,
        'sentiment_label': sentiment_label
    }

if __name__ == "__main__":
    myapp.add_command(import_headlines, name='import-headlines')
    myapp.add_command(extract_entities_and_analyze, name='extract-entities')
    myapp.add_command(top100entitieswithtype, name='top100entitieswithtype')
    myapp.add_command(allheadlinesfor, name='allheadlinesfor')
    myapp()
