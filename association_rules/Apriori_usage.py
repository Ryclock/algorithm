import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori

if __name__ == "__main__":
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    df = pd.DataFrame(
        data=np.zeros(shape=(len(dataset), 5), dtype=bool),
        columns=[1, 2, 3, 4, 5],
    )
    for i in range(len(dataset)):
        for j in dataset[i]:
            df.loc[i, j] = True
    frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
    print(pd.DataFrame(frequent_itemsets))
    # pd.DataFrame(frequent_itemsets).to_csv('frequent_itemsets.csv')
