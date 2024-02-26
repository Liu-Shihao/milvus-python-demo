from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility



DATASET = 'squad'  # Huggingface Dataset to use
MODEL = 'bert-base-uncased'  # Transformer to use for embeddings
TOKENIZATION_BATCH_SIZE = 1000  # Batch size for tokenizing operation
INFERENCE_BATCH_SIZE = 64  # batch size for transformer
INSERT_RATIO = .001  # How many titles to embed and insert
COLLECTION_NAME = 'huggingface_db'  # Collection name
DIMENSION = 768  # Embeddings size
LIMIT = 10  # How many results to search for

URI = "http://192.168.153.100:19530"
TOKEN = "root:Milvus"

# Load Collection
connections.connect(uri=URI, token=TOKEN)

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)

fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='original_question', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='answer', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='original_question_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
    'metric_type':'L2',
    'index_type':"IVF_FLAT",
    'params':{"nlist":1536}
}
collection.create_index(field_name="original_question_embedding", index_params=index_params)