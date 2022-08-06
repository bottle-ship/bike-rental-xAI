import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def main():
    df = pd.read_csv("../datasets/day.csv")
    df = df.drop(columns=["instant", "dteday", "casual", "registered"])

    pg = sns.PairGrid(df, corner=True, aspect=1.78)
    pg.map_diag(sns.histplot, hue=None, color="0.3")
    pg.map_lower(sns.scatterplot, alpha=0.3, edgecolor="none", c=df["cnt"].values)
    pg.tight_layout()

    for i, feature in enumerate(df.columns.tolist()):
        pg.axes[-1][i].set_xlabel(feature, fontdict={"fontsize": 25})
        pg.axes[i][0].set_ylabel(feature, fontdict={"fontsize": 25})

    plt.savefig("pairplot.png")
    plt.close()

    fig, ax = plt.subplots(1, 1)
    fig.canvas.set_window_title("Correlation Heatmap")
    df_corr = df.corr().round(decimals=3)
    mask = np.zeros_like(df_corr.values)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(
        df_corr,
        vmin=-1,
        vmax=1,
        # cmap=sns.diverging_palette(
        #     10, 250, s=100, l=50, sep=1, center="dark", as_cmap=True
        # ),
        # cmap=sns.color_palette("Spectral", as_cmap=True),
        cmap=sns.color_palette("RdBu", as_cmap=True),
        center=0.0,
        annot=True,
        annot_kws={"fontsize": 5.5},
        linewidths=0.2,
        square=False,
        xticklabels=df.columns.tolist(),
        yticklabels=df.columns.tolist(),
        mask=mask,
        ax=ax
    )
    ax.set_xticklabels(df.columns.tolist(), fontsize=8)
    ax.set_yticklabels(df.columns.tolist(), fontsize=8)
    ax.collections[0].colorbar.ax.tick_params(labelsize=8)
    ax.set_title("Correlation Heatmap", fontdict={"fontsize": 12}, pad=12)

    plt.tight_layout()
    plt.savefig("corr_heatmap.png")
    plt.close()


if __name__ == "__main__":
    main()
