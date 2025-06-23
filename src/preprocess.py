import pandas as pd


def load_mutation_data(mutation_path):
    df = pd.read_parquet(mutation_path)
    df = df[df['Variant_Type'].isin(['SNP', 'DEL', 'INS'])]
    df['Hugo_Symbol'] = df['Hugo_Symbol'].fillna('')
    df = df.rename(columns={'Start_position': 'Start', 'End_position': 'End'})
    return df


def load_expression_data(expression_path):
    df = pd.read_parquet(expression_path)
    return df.drop_duplicates(subset=['gene_name', 'case_barcode'])


def load_signatures_data(signatures_path):
    return pd.read_csv(signatures_path, sep='\t')