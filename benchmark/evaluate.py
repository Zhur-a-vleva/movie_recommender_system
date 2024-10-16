import argparse
import json
import tqdm
import os

import torch
import numpy as np
from torch_geometric.data import HeteroData

from matplotlib import pyplot as plt

from benchmark.metrics import compute_rmse, compute_pr_re, compute_ndcg


def prepare_predictions(part):
    test_data = HeteroData(_mapping=torch.load(f"../data/prepared/data{part}_test.pt"))

    user_number = test_data["user"].x.shape[0]
    movie_number = test_data["movie"].x.shape[0]

    test_edges = test_data["user", "rates", "movie"].edge_index
    full_edges = torch.zeros((2, user_number * movie_number))

    final_prediction = torch.load("../benchmark/full_predictions.pt")
    predictions = torch.zeros(test_edges.shape[1])

    for user in range(user_number):
        for movie in range(movie_number):
            full_edges[0][(user + 1) * (movie + 1) - 1] = user
            full_edges[1][(user + 1) * (movie + 1) - 1] = movie

    for i in range(test_edges.shape[1]):
        user, movie = test_edges[0][i], test_edges[1][i]
        index = (user + 1) * (movie + 1) - 1
        predictions[i] = final_prediction[index]

    return predictions, test_edges, test_data, user_number


def compute_metrics(predictions, test_edges, test_data, user_number, k):
    test_users = set(test_edges[0].tolist())

    precisions, recalls, NDCGs = [], [], []
    for user in range(user_number):
        if user not in test_users:
            # Only evaluate for users that are in test set
            continue

        user_links = (user == test_edges[0]).nonzero()
        model_movie_rating = torch.cat((test_edges[1][user_links], predictions[user_links]), dim=1)
        model_movie_rating = model_movie_rating[model_movie_rating[:, 1].sort(descending=True)[1]]

        recommendation = model_movie_rating[:k, 0].long()

        precision, recall = compute_pr_re(recommendation, test_data, user)
        ndcg = compute_ndcg(recommendation, test_data, user)

        precisions.append(precision)
        recalls.append(recall)
        NDCGs.append(ndcg)

    metrics = {
        "K": k,
        "Precision": np.mean(precisions),
        "Recall": np.mean(recalls),
        "NDCG": np.mean(NDCGs),
        "RMSE": compute_rmse(predictions, test_data["user", "rates", "movie"].edge_label)
    }
    return metrics


def visualize_metrics():
    filepaths = [
        "metrics/" + filename
        for _, _, filenames in os.walk("metrics") for filename in filenames
        if filename.split('.')[1] == "json"
    ]

    metrics = {"Precision": [], "Recall": [], "NDCG": [], "RMSE": []}
    part_names = []
    for filepath in filepaths:
        with open(filepath, "r") as metric_file:
            json_data = json.load(metric_file)
            [
                metrics[key].append(json_data[key])
                for key in metrics
            ]
            part_names.append(f"u{json_data['part']}.test, K={json_data['K']}")

    for i, metric in enumerate(metrics):
        fig = plt.figure(i)
        fig.set_size_inches(12, 5)
        plt.bar(part_names, metrics[metric])
        if metric != "RMSE":
            plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        plt.title(metric)
        plt.savefig(f"plots/{metric}_plot.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=20, help="Number of recommendations to evaluate")
    args = parser.parse_args()

    data_parts = ['1', '2', '3', '4', '5', 'a', 'b']
    total_metrics = {
        "K": args.k,
        "Precision": 0,
        "Recall": 0,
        "NDCG": 0,
        "RMSE": 0
    }

    for part in tqdm.tqdm(data_parts, desc="Data parts processed"):
        predictions, test_edges, test_data, user_number = prepare_predictions(part)

        metrics = compute_metrics(predictions, test_edges, test_data, user_number, args.k)

        total_metrics["Precision"] += metrics["Precision"]
        total_metrics["Recall"] += metrics["Recall"]
        total_metrics["NDCG"] += metrics["NDCG"]
        total_metrics["RMSE"] += metrics["RMSE"]

        # Save calculated metrics to metric file
        with open(f"metrics/metrics_for_part_{part}.json", "w") as metric_file:
            metrics["part"] = part
            metric_file.write(json.dumps(metrics))

    visualize_metrics()

    total_metrics["Precision"] /= len(data_parts)
    total_metrics["Recall"] /= len(data_parts)
    total_metrics["NDCG"] /= len(data_parts)
    total_metrics["RMSE"] /= len(data_parts)

    print("Averaged metrics:")
    print(f"Precision@{args.k} = {total_metrics['Precision']}")
    print(f"Recall@{args.k} = {total_metrics['Recall']}")
    print(f"NDCG@{args.k} = {total_metrics['NDCG']}")
    print(f"RMSE = {total_metrics['RMSE']}")
