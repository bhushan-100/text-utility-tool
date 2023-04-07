import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess(articles, summaries, categories):
    df = pd.DataFrame(
        {'articles': articles, 'summaries': summaries, 'categories': categories},)
    df = df[['articles', 'summaries']]

    # -- get length of each article and summary for analysis
    df['articles_length'] = df['articles'].apply(lambda x: len(x.split()))
    df['summaries_length'] = df['summaries'].apply(lambda x: len(x.split()))
    # print(df)

    df['articles'] = df['articles'].str.encode(
        'ascii', 'ignore').str.decode('ascii')
    df['summaries'] = df['summaries'].str.encode(
        'ascii', 'ignore').str.decode('ascii')

    df.dropna()
    return df


def split_data(df):
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df, val_df = train_test_split(
        train_df, test_size=0.2, random_state=42)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    return train_df, val_df, test_df
