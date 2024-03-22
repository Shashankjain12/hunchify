import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

from IPython.display import display


def convert_df_to_dict(
    df,
    user_col_name="user_code",
    item_col_name="poll_code",
    rating_col_name="event_score",
    with_pred_rating=True,
):
    if with_pred_rating:
        user_item_dict = (
            df.sort_values(rating_col_name, ascending=False)
            .groupby(user_col_name)
            .apply(lambda x: list(zip(x[item_col_name], x[rating_col_name])))
            .to_dict()
        )
    else:
        user_item_dict = df.groupby(user_col_name)[item_col_name].apply(list).to_dict()

    return user_item_dict


def precision(recommended, actual, with_pred_rating=True):
    precision_dict = {}
    if with_pred_rating:
        for user in actual.keys():
            actual_set = set([item for item, rating in actual[user]])
            recommended_set = set([item for item, rating in recommended[user]])
            precision_dict[user] = (
                len(actual_set & recommended_set) / len(recommended_set) if recommended_set else 0
            )
    else:
        for user in actual.keys():
            actual_set = set([item for item, rating in actual[user]])
            recommended_set = set(recommended[user])
            precision_dict[user] = (
                len(actual_set & recommended_set) / len(recommended_set) if recommended_set else 0
            )

    return precision_dict


def recall(recommended, actual, with_pred_rating=True):
    recall_dict = {}
    if with_pred_rating:
        for user in actual.keys():
            actual_set = set([item for item, rating in actual[user]])
            recommended_set = set([item for item, rating in recommended[user]])
            recall_dict[user] = (
                len(actual_set & recommended_set) / len(actual_set) if actual_set else 0
            )
    else:
        for user in actual.keys():
            actual_set = set([item for item, rating in actual[user]])
            recommended_set = set(recommended[user])
            recall_dict[user] = (
                len(actual_set & recommended_set) / len(actual_set) if actual_set else 0
            )

    return recall_dict


def ndcg_without_pred_rating(recommended_items_with_score, true_items_with_scores):
    ndcg_scores = {}

    for user, recommended_items in recommended_items_with_score.items():
        true_items = dict(true_items_with_scores[user])

        dcg = 0
        for i, item in enumerate(recommended_items):
            if item in true_items:
                dcg += true_items[item] / np.log2(i + 2)

        idcg = 0
        for i, (_, true_score) in enumerate(true_items.items()):
            idcg += true_score / np.log2(i + 2)

        ndcg_scores[user] = dcg / idcg if idcg > 0 else 0

    return ndcg_scores


def ndcg_with_pred_rating(recommended_items_with_score, true_items_with_scores):
    ndcg_scores = {}

    for user, recommended_items in recommended_items_with_score.items():
        true_items = dict(true_items_with_scores[user])

        dcg = 0
        for i, (item, pred_score) in enumerate(recommended_items):
            if item in true_items:
                dcg += pred_score / np.log2(i + 2)

        idcg = 0
        for i, (_, true_score) in enumerate(true_items.items()):
            idcg += true_score / np.log2(i + 2)

        ndcg_scores[user] = dcg / idcg if idcg > 0 else 0

    return ndcg_scores


def product_coverage(recommended_items_with_score, all_items, with_pred_rating=True):
    if with_pred_rating:
        recommended_items = [
            item
            for recommendations in recommended_items_with_score.values()
            for item, rating in recommendations
        ]
    else:
        recommended_items = [
            item
            for recommendations in recommended_items_with_score.values()
            for item in recommendations
        ]

    unique_recommended_items = set(recommended_items)

    coverage = len(unique_recommended_items) / len(all_items)

    return coverage


def personalization(recommended_items_with_score, with_pred_rating=True):
    if with_pred_rating:
        recommended_items = [
            item
            for recommendations in recommended_items_with_score.values()
            for item, rating in recommendations
        ]
    else:
        recommended_items = [
            item
            for recommendations in recommended_items_with_score.values()
            for item in recommendations
        ]

    unique_recommended_items = list(set(recommended_items))

    item_to_index = {item: index for index, item in enumerate(unique_recommended_items)}

    matrix = np.zeros((len(recommended_items_with_score), len(unique_recommended_items)))

    if with_pred_rating:
        for user_index, recommendations in enumerate(recommended_items_with_score.values()):
            for item, rating in recommendations:
                matrix[user_index, item_to_index[item]] = 1
    else:
        for user_index, recommendations in enumerate(recommended_items_with_score.values()):
            for item in recommendations:
                matrix[user_index, item_to_index[item]] = 1

    jaccard_distances = pairwise_distances(matrix.astype(bool), metric="jaccard")

    dissimilarity = np.mean(jaccard_distances)

    return dissimilarity


def novelty(recommended_items_with_score, df, item_col_name="poll_code", with_pred_rating=True):
    item_popularity = df[item_col_name].value_counts()

    item_rank = item_popularity.rank(method="min", ascending=False)

    if with_pred_rating:
        recommended_item_ranks = [
            item_rank[item]
            for user, recommendations in recommended_items_with_score.items()
            for item, rating in recommendations
        ]
    else:
        recommended_item_ranks = [
            item_rank[item]
            for user, recommendations in recommended_items_with_score.items()
            for item in recommendations
        ]
    average_rank = np.mean(recommended_item_ranks)

    novelty = average_rank / len(item_rank)

    return novelty


def evaluate(
    recommended_items_with_score,
    true_items_with_scores,
    all_items,
    df,
    item_col_name="poll_code",
    with_pred_rating=True,
):
    if with_pred_rating:
        ndcg_by_user = ndcg_with_pred_rating(recommended_items_with_score, true_items_with_scores)
    else:
        ndcg_by_user = ndcg_without_pred_rating(
            recommended_items_with_score, true_items_with_scores
        )
    precision_by_user = precision(
        recommended_items_with_score, true_items_with_scores, with_pred_rating
    )
    recall_by_user = recall(recommended_items_with_score, true_items_with_scores, with_pred_rating)
    product_coverage_score = product_coverage(
        recommended_items_with_score, all_items, with_pred_rating
    )
    personalization_score = personalization(recommended_items_with_score, with_pred_rating)
    novelty_score = novelty(recommended_items_with_score, df, item_col_name, with_pred_rating)

    results = {
        "NDCG_median": np.median(list(ndcg_by_user.values())),
        "NDCG_mean": np.mean(list(ndcg_by_user.values())),
        "NDCG_std": np.std(list(ndcg_by_user.values())),
        "precision_median": np.median(list(precision_by_user.values())),
        "precision_mean": np.mean(list(precision_by_user.values())),
        "precision_std": np.std(list(precision_by_user.values())),
        "recall_median": np.median(list(recall_by_user.values())),
        "recall_mean": np.mean(list(recall_by_user.values())),
        "recall_std": np.std(list(recall_by_user.values())),
        "product_coverage": product_coverage_score,
        "personalization": personalization_score,
        "novelty": novelty_score,
    }

    return ndcg_by_user, precision_by_user, recall_by_user, results


def eval_add_show(
    model_name,
    recommendation_dict,
    actual_dict,
    all_poll_codes,
    train_data,
    item_col_name="poll_code",
    with_pred_rating=True,
    model_results_comparison=None,
    add=True,
    show=True,
):
    ndcg_by_user, precision_by_user, recall_by_user, results = evaluate(
        recommendation_dict,
        actual_dict,
        all_poll_codes,
        train_data,
        item_col_name=item_col_name,
        with_pred_rating=with_pred_rating,
    )
    if (add) and (model_results_comparison is not None):
        model_results_comparison = pd.concat(
            [model_results_comparison, pd.DataFrame([results], index=[model_name])],
            ignore_index=False,
        )
        model_results_comparison = model_results_comparison[
            ~model_results_comparison.index.duplicated(keep="last")
        ]
    if show:
        # df = pd.DataFrame()
        # for metric, s in zip(
        #     ["ndcg", "precison", "recall"],
        #     [ndcg_by_user, precision_by_user, recall_by_user],
        # ):
        #     df[metric] = pd.Series(list(s.values())).describe()
        # with pd.option_context("display.float_format", "{:,.2%}".format):
        #     display(df.transpose().iloc[:, 1:])
        with pd.option_context("display.float_format", "{:,.2%}".format):
            display(model_results_comparison.loc[model_name].to_frame().transpose())

    return ndcg_by_user, precision_by_user, recall_by_user, results, model_results_comparison


def main():
    actual_df = pd.DataFrame(
        {
            "user": [1, 1, 1, 1, 1, 2, 2, 3, 3, 3, 3, 4],
            "item": [1, 2, 3, 4, 5, 2, 8, 2, 6, 7, 5, 1],
            "rating": [0.8, 0.6, 0.4, 0.2, 0.1, 0.7, 0.5, 0.9, 0.6, 0.4, 0.2, 0.8],
        }
    )

    true_items_with_scores = convert_df_to_dict(
        actual_df, user_col_name="user", item_col_name="item", rating_col_name="rating"
    )

    true_items_without_scores = convert_df_to_dict(
        actual_df, user_col_name="user", item_col_name="item", with_pred_rating=False
    )

    all_items = actual_df.item.unique().tolist()

    recommended_df = pd.DataFrame(
        {
            "user": [1, 1, 1, 2, 2, 3, 3, 3, 4],
            "item": [1, 2, 3, 2, 8, 2, 6, 7, 1],
            "rating": [0.2, 0.1, 0.9, 0.5, 0.6, 0.7, 0.5, 0.2, 0.8],
        }
    )

    recommended_items_with_scores = convert_df_to_dict(
        recommended_df, user_col_name="user", item_col_name="item", rating_col_name="rating"
    )

    recommended_items_without_scores = convert_df_to_dict(
        recommended_df, user_col_name="user", item_col_name="item", with_pred_rating=False
    )

    print(
        precision(recommended_items_without_scores, true_items_with_scores, with_pred_rating=False)
    )

    print(recall(recommended_items_without_scores, true_items_with_scores, with_pred_rating=False))

    ndcg_scores_without_pred_rating = ndcg_without_pred_rating(
        recommended_items_with_scores, true_items_with_scores
    )
    print(ndcg_scores_without_pred_rating)

    ndcg_scores_with_pred_rating = ndcg_with_pred_rating(
        recommended_items_with_scores, true_items_with_scores
    )
    print(ndcg_scores_with_pred_rating)

    print(product_coverage(recommended_items_with_scores, all_items))

    print(personalization(recommended_items_with_scores))

    print(novelty(recommended_items_with_scores, actual_df, "item"))


if __name__ == "__main__":
    main()
