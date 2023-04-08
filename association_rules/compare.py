import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.preprocessing import TransactionEncoder
import timeit


if __name__ == "__main__":
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    te = TransactionEncoder()
    te.fit(dataset)
    df = pd.DataFrame(data=te.transform(dataset), columns=te.columns_)
    trials = 100
    repeat = 10
    output = '{:s} - {} trials: [min time {:.2f}s, max time {:.2f}s, spread {:.2f}%]'
    t = timeit.repeat(
        "apriori(df, min_support=0.6, use_colnames=True)", number=trials, repeat=repeat, globals=locals())
    print(output.format('apriori', trials, min(t),
          max(t), 100*(max(t) - min(t))/min(t)))
    t = timeit.repeat(
        "apriori(df, min_support=0.6, use_colnames=True, low_memory=True)", number=trials, repeat=repeat, globals=locals())
    print(output.format('apriori_low_memory', trials, min(t),
          max(t), 100*(max(t) - min(t))/min(t)))
    t = timeit.repeat(
        "fpgrowth(df, min_support=0.6, use_colnames=True)", number=trials, repeat=repeat, globals=locals())
    print(output.format('fpgrowth', trials, min(t),
          max(t), 100*(max(t) - min(t))/min(t)))
