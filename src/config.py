import random
import numpy as np
import tensorflow as tf
import kagglehub 
import os
import urllib.request
import gzip
import shutil

signature_cols = ['SBS1', 'SBS2', 'SBS3', 'SBS4', 'SBS5',
    'SBS6', 'SBS7a', 'SBS7b', 'SBS7c', 'SBS7d', 'SBS8', 'SBS9', 'SBS10a',
    'SBS10b', 'SBS10c', 'SBS10d', 'SBS11', 'SBS12', 'SBS13', 'SBS14',
    'SBS15', 'SBS16', 'SBS17a', 'SBS17b', 'SBS18', 'SBS19', 'SBS20',
    'SBS21', 'SBS22a', 'SBS22b', 'SBS23', 'SBS24', 'SBS25', 'SBS26',
    'SBS27', 'SBS28', 'SBS29', 'SBS30', 'SBS31', 'SBS32', 'SBS33', 'SBS34',
    'SBS35', 'SBS36', 'SBS37', 'SBS38', 'SBS39', 'SBS40a', 'SBS40b',
    'SBS40c', 'SBS41', 'SBS42', 'SBS43', 'SBS44', 'SBS45', 'SBS46', 'SBS47',
    'SBS48', 'SBS49', 'SBS50', 'SBS51', 'SBS52', 'SBS53', 'SBS54', 'SBS55',
    'SBS56', 'SBS57', 'SBS58', 'SBS59', 'SBS60', 'SBS84', 'SBS85', 'SBS86',
    'SBS87', 'SBS88', 'SBS89', 'SBS90', 'SBS91', 'SBS92', 'SBS93', 'SBS94',
    'SBS95', 'SBS96', 'SBS97', 'SBS98', 'SBS99']


class_short_labels = {
    "Hematopoietic and reticuloendothelial systems": "Hemato",
    "Bronchus and lung": "Lung",
    "Breast": "Breast",
    "Kidney": "Kidney",
    "Brain": "Brain",
    "Colon": "Colon",
    "Corpus uteri": "Corpus",
    "Skin": "Skin",
    "Prostate gland": "Prostate",
    "Stomach": "Stomach",
    "Bladder": "Bladder",
    "Liver and intrahepatic bile ducts": "Liver",
    "Pancreas": "Pancreas",
    "Ovary": "Ovary",
    "Uterus, NOS": "Uterus",
    "Cervix uteri": "Cervix",
    "Esophagus": "Esophagus",
    "Adrenal gland": "Adrenal",
    "Other and ill-defined sites": "Other",
    "Other and unspecified parts of tongue": "Tongue",
    "Connective, subcutaneous and other soft tissues": "Connective",
    "Larynx": "Larynx",
    "Rectum": "Rectum",
    "Other and ill-defined sites in lip, oral cavity and pharynx": "Oral/Pharynx"
}


def download_kaggle_data():
    """Pobiera dane z Kaggle używając kagglehub"""
    print("Pobieranie danych z Kaggle...")
    dataset_path = kagglehub.dataset_download("martininf1n1ty/oncokb-cancer-gene-list")
    print(f"Dane pobrane do: {dataset_path}")
    return dataset_path


def download_hg38_reference():
    """Pobiera genom referencyjny hg38 z UCSC"""
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    fasta_path = os.path.join(data_dir, 'hg38.fa')
    fasta_gz_path = fasta_path + '.gz'
    
    # Jeśli plik już istnieje, nie pobieraj ponownie
    if os.path.exists(fasta_path):
        print(f"Plik hg38.fa już istnieje w {fasta_path}")
        return fasta_path
    
    # Pobierz spakowany plik
    if not os.path.exists(fasta_gz_path):
        print("Pobieranie hg38.fa.gz z UCSC (to może chwilę potrwać - plik jest duży)...")
        url = "http://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
        urllib.request.urlretrieve(url, fasta_gz_path)
        print(f"Pobrano do {fasta_gz_path}")
    
    # Rozpakuj plik
    print("Rozpakowywanie hg38.fa.gz...")
    with gzip.open(fasta_gz_path, 'rb') as f_in:
        with open(fasta_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    print(f"Rozpakowano do {fasta_path}")
    
    # Opcjonalnie usuń spakowany plik
    # os.remove(fasta_gz_path)
    
    return fasta_path


# Pobierz dane z Kaggle
kaggle_data_path = download_kaggle_data()

# Pobierz genom referencyjny hg38
hg38_path = download_hg38_reference()

data_paths = {
    'mutations': os.path.join(kaggle_data_path, 'mutations_3.parquet'),
    'expression': os.path.join(kaggle_data_path, 'expression_2.parquet'),
    'signatures': os.path.join(kaggle_data_path, 'Assignment_Solution_Activities.txt'),
    'reference_fasta': hg38_path
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)