import chromadb

# connect to DB
client = chromadb.PersistentClient(path="webcam_face_db")
collection = client.get_collection("face_embeddings")

data = collection.get()

for item in data["metadatas"]:
    print(item)
    data = collection.get()

print("Total records:", len(data["ids"]))
# get all data
data = collection.get()

print("Total records:", len(data["ids"]))

# print all metadata
for item in data["metadatas"]:
    print(item)