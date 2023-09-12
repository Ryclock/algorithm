import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder

if __name__ == "__main__":
    dataset = [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
    te = TransactionEncoder()
    te.fit(dataset)
    df = pd.DataFrame(data=te.transform(dataset), columns=te.columns_)
    frequent_itemsets = apriori(df, min_support=0.4, use_colnames=True)
    print(pd.DataFrame(frequent_itemsets))
