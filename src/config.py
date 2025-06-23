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

data_paths = {
    'mutations': 'data/mutations_3.parquet',
    'expression': 'data/expression_2.parquet',
    'signatures': 'data/Assignment_Solution_Activities.txt',
    'reference_fasta': 'data/hg38.fa'
}
