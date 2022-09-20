import pandas as pd


def pair_plot_sample(df, seg, sample_size):
    df_lst = list()

    for s in df[seg].unique():
        df_lst.append(df[df[seg] == s].sample(sample_size, replace=True))

    return pd.concat(df_lst)
