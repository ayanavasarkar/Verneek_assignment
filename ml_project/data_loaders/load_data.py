import pandas as pd
import os
from typing import List, Tuple
import numpy as np


def load_articles(data_folder: str) -> pd.DataFrame:
    """
    Load the articles as pandas dataframe

    Keyword arguments:
        data_folder -- the path to the folder that contains the data

    Returns:
        pandas dataframe containing the article data
    """
    return pd.read_csv(
        open(os.path.join(data_folder, "raw-data.csv"), encoding="latin_1")
    )


def load_tags(data_folder: str) -> List[str]:
    """
    Load the tag names as list

    Keyword arguments:
        data_folder -- the path to the folder that contains the data

    Returns:
        list of tag names. The index of the tags in the list are the ids of the tags
    """
    tags = []
    with open(os.path.join(data_folder, "tags.dat")) as tag_file:
        for tag in tag_file.readlines():
            tags.append(tag.strip())
    return tags


def load_article_tags(data_folder: str) -> List[List[str]]:
    """
    Load the tags for each article

    Keyword arguments:
        data_folder -- the path to the folder that contains the data

    Returns:
        list of tags per article. This is represented as a 2d list. Each element of the outer list
        represents the article at that index in the pandas Dataframe. The inner lists contain the
        tag ids for each article. So to get the first tag name for the first article you would call:
        tags[article_tags[0][0]]
    """
    item_tag = []
    with open(os.path.join(data_folder, "item-tag.dat")) as item_tag_file:
        for i, line in enumerate(item_tag_file.readlines()):
            item_tag.append([int(tag) for tag in line.strip().split()[1:]])
    return item_tag


def load_citations(data_folder: str, num_articles: int = 16980) -> np.ndarray:
    """
    Load the citation network

    Keyword arguments:
        data_folder -- the path to the folder that contains the data
        num_articles -- the number of articles in the dataset

    Returns:
        square 2d numpy matrix with citation information about articles.
        The citation matrix is num_articles X num_articles.
        There is a 1 in locations in which there is a citation between articles a -> b
        citations[a][b] and 0 otherwise.
    """
    citations = np.zeros([num_articles, num_articles])
    with open(os.path.join(data_folder, "citations.dat")) as citations_file:
        for i, line in enumerate(citations_file.readlines()):
            for citation in line.strip().split()[1:]:
                citations[i][int(citation)] = 1
    return citations


def load_user_article_likes(
    data_folder: str, num_users: int = 5551, num_articles: int = 13584
) -> np.ndarray:
    """
    Load the article likes for each user

    Keyword arguments:
        data_folder -- the path to the folder that contains the data
        num_users -- the number of users in the dataset - default is 5551
        num_articles -- the number of articles in the dataset - default is 13584

    Returns:
        2d numpy matrix with describing user like information.
        This matrix is num_users X num_articles.
        There is a 1 in locations in which a user likes an article u -> a
        citations[u][a] and 0 otherwise.
    """
    user_items = np.zeros([num_users, num_articles])
    with open(os.path.join(data_folder, "train_data.dat")) as users_file:
        for i, line in enumerate(users_file.readlines()):
            for article in line.strip().split()[1:]:
                user_items[i][int(article)] = 1
    return user_items


def load_articles_and_user_article_likes(
    data_folder: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load both the article dataframe and the user article likes numpy array.

    Keyword arguments:
        data_folder -- the path to the folder that contains the data

    Returns:
        Tuple
        First element is a dataframe containing article info
        Second is 2d numpy matrix with describing user like information.
    """
    articles_df = load_articles(data_folder)
    user_article_likes = load_user_article_likes(data_folder)
    return (articles_df, user_article_likes)
