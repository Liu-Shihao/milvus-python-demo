from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from datasets import load_dataset_builder, load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from torch import clamp, sum


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

connections.connect(uri=URI, token=TOKEN)
fields = [
    FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name='original_question', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='answer', dtype=DataType.VARCHAR, max_length=1000),
    FieldSchema(name='original_question_embedding', dtype=DataType.FLOAT_VECTOR, dim=DIMENSION)
]
schema = CollectionSchema(fields=fields)
collection = Collection(name=COLLECTION_NAME, schema=schema)
collection.load()

tokenizer = AutoTokenizer.from_pretrained(MODEL)
def tokenize_question(batch):
    results = tokenizer(batch['question'], add_special_tokens = True, truncation = True, padding = "max_length", return_attention_mask = True, return_tensors = "pt")
    batch['input_ids'] = results['input_ids']
    batch['token_type_ids'] = results['token_type_ids']
    batch['attention_mask'] = results['attention_mask']
    return batch

model = AutoModel.from_pretrained(MODEL)
def embed(batch):
    sentence_embs = model(
                input_ids=batch['input_ids'],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask']
                )[0]
    input_mask_expanded = batch['attention_mask'].unsqueeze(-1).expand(sentence_embs.size()).float()
    batch['question_embedding'] = sum(sentence_embs * input_mask_expanded, 1) / clamp(input_mask_expanded.sum(1), min=1e-9)
    return batch

questions = {'question':['When was chemistry invented?', 'When was Eisenhower born?']}
question_dataset = Dataset.from_dict(questions)

question_dataset = question_dataset.map(tokenize_question, batched = True, batch_size=TOKENIZATION_BATCH_SIZE)
question_dataset.set_format('torch', columns=['input_ids', 'token_type_ids', 'attention_mask'], output_all_columns=True)
question_dataset = question_dataset.map(embed, remove_columns=['input_ids', 'token_type_ids', 'attention_mask'], batched = True, batch_size=INFERENCE_BATCH_SIZE)

def search(batch):
    res = collection.search(batch['question_embedding'].tolist(), anns_field='original_question_embedding', param = {}, output_fields=['answer', 'original_question'], limit = LIMIT)
    overall_id = []
    overall_distance = []
    overall_answer = []
    overall_original_question = []
    for hits in res:
        ids = []
        distance = []
        answer = []
        original_question = []
        for hit in hits:
            ids.append(hit.id)
            distance.append(hit.distance)
            answer.append(hit.entity.get('answer'))
            original_question.append(hit.entity.get('original_question'))
        overall_id.append(ids)
        overall_distance.append(distance)
        overall_answer.append(answer)
        overall_original_question.append(original_question)
    return {
        'id': overall_id,
        'distance': overall_distance,
        'answer': overall_answer,
        'original_question': overall_original_question
    }
question_dataset = question_dataset.map(search, batched=True, batch_size = 1)
for x in question_dataset:
    print()
    print('Question:')
    print(x['question'])
    print('Answer, Distance, Original Question')
    for x in zip(x['answer'], x['distance'], x['original_question']):
        print(x)
