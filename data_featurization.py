import pandas as pd
import numpy as np
import json
import os


def factorize_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    # embed missing data as it's own feature, can't be -1 as we need to look up it's embedding index later in the model
    na_sentinel = len(pd.unique(df[col])) - 1
    factorizer = pd.factorize(df[col], na_sentinel=na_sentinel)
    map = dict(zip(np.arange(len(factorizer[1])).tolist(), factorizer[1]))
    if na_sentinel in factorizer[0]:
        map[len(map)] = na_sentinel
    df[col] = factorizer[0]

    if not os.path.isdir("data"):
        os.mkdir("data")
    with open(f"data/{col}_map.json", "w") as f:
        json.dump(map, f)
    return df


def main():
    embedding_cols = []
    df = pd.read_csv("LoanData.csv", skiprows=1)
    df = df.drop(["desc", "url"], axis=1)
    for col in df.columns:
        if df[col].dtype == object:
            try:
                # see if we can turn into datetime and base our delta on days from issue_date
                df[col] = (
                    pd.to_datetime(df[col]) - pd.to_datetime(df["issue_d"])
                ).dt.days.fillna(-1)
            except ValueError:
                if (
                    df[col].str.contains("%")[0]
                    and type(df[col].str.contains("%")[0]) != float
                ):
                    df[col] = df[col].str.replace("%", "").astype(float)
                else:
                    # otherwise treat as categorical and treat as embedding
                    df = factorize_column(df, col)
                    embedding_cols.append(col)
        elif df[col].dtype == float:
            # float can contain nan
            df[col] = df[col].fillna(-1)

    for col in df.columns:
        assert df[col].dtype in [int, float]

    df.to_csv("data/LoanData_Numeric.csv")
    with open("data/embedding_cols.txt", "w") as f:
        for col in embedding_cols:
            f.write(col + "\n")


if __name__ == "__main__":
    main()
