
from sentence_transformers import SentenceTransformer, util

def get_semantic_similarity(sentence1,sentence2):

    
    model = SentenceTransformer('./EmbeddingModel/sentence-transformers/all_minilm_l6_v2')
    
    embedding1 = model.encode(sentence1, convert_to_tensor=True)
    embedding2 = model.encode(sentence2, convert_to_tensor=True)

    
    cosine_similarity = util.cos_sim(embedding1, embedding2)
    
    float_value = (cosine_similarity.item() + 1) / 2

    return float_value


if __name__=="__main__":
    sentence1 = "its a good day"
    sentence2 = "hello how are you"
    print(get_semantic_similarity(sentence1,sentence2))