import tqdm


def recall_at_m(test_data_mat, prediction_mat, M=50) -> float:
    """
    Calculate recall at M metric. This metric calculates the top M article
    like predictions for each user and counts the number of actual user
    predictions in that list. The metric per user is:
         # counts in top M / number of total likes
    we then average over all user metrics
    We skip over all users that don't have any likes.

    Keyword arguments:
        test_data_mat -- The test data. A user x article length matrix of 0s and 1s
        prediction_mat -- The prediction data. A user x article length matrix of user like scores
            a larger score indicates higher likelihood of user liking an articleself.
        M -- The number of top predictions to evaluate from.

    Returns:
        float -- the recall_at_m metric
    """
    assert (
        test_data_mat.shape == prediction_mat.shape
    ), "test matrix and prediction matrix need to have the same shape"
    user_likes = [0 for _ in range(test_data_mat.shape[0])]
    user_corrects = [0 for _ in range(test_data_mat.shape[0])]
    user_recall = [0 for _ in range(test_data_mat.shape[0])]
    non_zero_users = 0
    for user in tqdm(range(0, test_data_mat.shape[0])):
        user_likes[user] = sum(test_data_mat[user])
        ranked = sorted(
            enumerate(prediction_mat[user]), key=lambda x: x[1], reverse=True
        )[0:M]
        ranked_index = [i[0] for i in ranked]
        corrects = [
            i[0]
            for i in enumerate(test_data_mat[user])
            if i[0] in ranked_index and i[1] == 1
        ]
        user_corrects[user] = len(corrects)
        user_recall[user] = (
            user_corrects[user] / user_likes[user] if user_likes[user] > 0 else 0.0
        )
        if user_likes[user] > 0:
            non_zero_users += 1
    total_recall = sum(user_recall) / non_zero_users
    return total_recall
