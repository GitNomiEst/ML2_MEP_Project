import requests
from pymongo import MongoClient

def get_neo_data(api_key):

    all_data = []
    x = 0

    base_url = "https://api.nasa.gov/neo/rest/v1/feed"
    next_url = base_url
    print(next_url)
    while x<=10:
        x+=1
        print("entries loaded.. continues")
        response = requests.get(next_url, params={"api_key": api_key})
        print(response.status_code)
        if response.status_code == 200:
            neo_data = response.json()
            all_data.append(neo_data)
            
            # Get the link for the next page, if it exists
            next_url = neo_data.get("links", {}).get("next")
        else:
            print(f"Error: {response.status_code}")
            return None

    print("All NEO data loaded.")
    return all_data


def save_to_mongodb(data):
    if data is not None and isinstance(data, list) and data:
        # Initialize MongoDB client with appropriate connection string
        client = MongoClient(f"mongodb+srv://nomi:***@nasa.mongocluster.cosmos.azure.com/?tls=true&authMechanism=SCRAM-SHA-256&retrywrites=false&maxIdleTimeMS=120000")
        
        db = client['nasa']
        collection = db['nasa']
        collection.insert_many(data)
        print("Data saved to MongoDB.")
    else:
        print("Could not save data to MongoDB.")


if __name__ == "__main__":
    api_key = "..." #To Do before run: enter API Key here
    neo_data = get_neo_data(api_key)
    save_to_mongodb(neo_data)