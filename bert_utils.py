import torch


def get_token_embeddings(encoded_layers):
    # convert [layers, batches, tokens, features] to [tokens, layers, features]
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1, 0, 2)
    return token_embeddings


def get_sum_word_vecs(token_embeddings):
    token_vecs_sum = []
    for token in token_embeddings:

        # OPTION
        sum_vec = torch.sum(token[-4:], dim=0)  # three last layers
        # sum_vec = torch.sum(token[-1], dim=0)
        
        token_vecs_sum.append(sum_vec)
    return token_vecs_sum


def get_sentence_vec(encoded_layers):
    # this sentence vec is just the average of all embedded tokens
    # TODO remove [CLS], [SEP] or '.' etc.
    token_vecs = encoded_layers[-1][0]  # last_layer, batch_id:0

    # TODO tf-idf
    sentence_embedding = torch.mean(token_vecs, dim=0)
    return sentence_embedding


def cos_similarity(vec1, vec2):
    dot = torch.dot(vec1, vec2)
    norm1 = torch.norm(vec1)
    norm2 = torch.norm(vec2)
    return (dot / (norm1 * norm2)).item()
