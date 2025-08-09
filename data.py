import pandas as pd
from pandas import DataFrame


def create_data_frame(data_file_name: str):
    return pd.read_csv(data_file_name, sep='\t')

def parse_data(data: DataFrame):
    for index, row in data.iterrows():
        pass     

df = create_data_frame('prediction.csv')
print(df.to_string())


import pandas as pd
import gzip


rna_data = pd.read_csv('Hackathon2024.RNA.txt.gz', sep='\t', index_col=0)


atac_data = pd.read_csv('Hackathon2024.ATAC.txt.gz', sep='\t', index_col=0)


train_pairs = pd.read_csv('Hackathon2024.Training.Set.Peak2Gene.Pairs.txt.gz', sep='\t')


test_pairs = pd.read_csv('Hackathon2024.Testing.Set.Peak2Gene.Pairs.txt.gz', sep='\t')


meta_data = pd.read_csv('Hackathon2024.Meta.txt.gz', sep='\t')


"""
Step 2:
Go through data, make sure its clean, no empty values
"""