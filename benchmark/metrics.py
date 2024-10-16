import numpy as np
from sklearn.metrics import mean_squared_error
import torch


def extract_movies_with_ratings_by_user(graph, user_id):
    edge_index = graph["user", "rates", "movie"].edge_index
    edge_label = graph["user", "rates", "movie"].edge_label

    indices = (user_id == graph["user", "rates", "movie"].edge_index[0]).nonzero()

    movie_ratings = torch.cat((edge_index[1][indices], edge_label[indices]), dim=1).long()

    return movie_ratings


def compute_rmse(predicted_ratings, true_rating):
    return np.sqrt(mean_squared_error(predicted_ratings, true_rating))


def compute_pr_re(recommendation, test_graph, user_id):
    tp, fp, fn = 0, 0, 0

    movie_ratings = extract_movies_with_ratings_by_user(test_graph, user_id)
    for movie in movie_ratings:
        if movie[1] >= 4:
            if movie[0] in recommendation:
                tp += 1
            else:
                fn += 1
        else:
            if movie[0] in recommendation:
                fp += 1
    return (tp / (tp + fp),
            tp / (tp + fn)) if tp > 0 else (0, 0)


def compute_dcg(recommendation, movie_ratings):
    dcg = 0
    for i, movie in enumerate(recommendation):
        if not movie in movie_ratings[:, 0]:
            score = 0
        else:
            indx = (movie_ratings[:, 0] == movie).nonzero().item()
            score = (2 ** (movie_ratings[indx][1]) - 1).item()
        discount = np.log2(i + 2)
        dcg += score / discount

    return dcg


def compute_ndcg(recommendation, test_graph, user_id):
    movie_ratings = extract_movies_with_ratings_by_user(test_graph, user_id)

    dcg = compute_dcg(recommendation, movie_ratings)

    best_recommendation = movie_ratings[movie_ratings[:, 1].sort(descending=True)[1]][:, 0]
    idcg = compute_dcg(best_recommendation, movie_ratings)

    return dcg / idcg
