import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess(articles, summaries, categories):
    df = pd.DataFrame(
        {'articles': articles, 'summaries': summaries, 'categories': categories})

    category_sizes = df.groupby('categories').size()
    sns.barplot(x=category_sizes.index, y=category_sizes)
    plt.show()

    # -- get length of each article and summary for analysis
    df['articles_length'] = df['articles'].apply(lambda x: len(x.split()))
    df['summaries_length'] = df['summaries'].apply(lambda x: len(x.split()))
    # print(df)

    category_length = df.groupby('categories', 0).agg(
        {"articles_length": "mean", "summaries_length": "mean"})
    df_m = pd.melt(category_length, ignore_index=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=df_m.index, y="value", hue="variable", data=df_m)
    plt.show()

    pd.melt(category_length, ignore_index=False).groupby('variable').mean()

    df = df[['articles', 'summaries']]

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
