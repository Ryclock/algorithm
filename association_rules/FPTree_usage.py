import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder

if __name__ == "__main__":
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    te = TransactionEncoder()
    te.fit(dataset)
    df = pd.DataFrame(data=te.transform(dataset), columns=te.columns_)
    frequent_itemsets = fpgrowth(df, min_support=0.6, use_colnames=True)
    print(frequent_itemsets.sort_values(by=['support'],ascending=False))
