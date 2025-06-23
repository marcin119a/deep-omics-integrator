import re
import pandas as pd


def get_cosmic_notation(row, fasta_dict):
    chrom = row.Chromosome
    start = int(row.Start)
    end = int(row.End)
    alt = str(row.Tumor_Seq_Allele2) if pd.notnull(row.Tumor_Seq_Allele2) else None
    variant_type = row.Variant_Type
    try:
        seq = fasta_dict[chrom].seq
        context_seq = seq[start-2:end+1].upper()
        ref_base = seq[start-1:end].upper()

        if variant_type == 'SNP':
            return f"{context_seq[0]}[{ref_base}>{alt}]{context_seq[-1]}"
        elif variant_type == 'DEL':
            deleted_seq = ref_base if end <= start else seq[start-1:end].upper()
            return f"{context_seq[0]}[del{len(deleted_seq)}]{context_seq[-1]}"
        elif variant_type == 'INS':
            return f"{context_seq[0]}[ins{len(alt)}]{context_seq[-1]}"
    except Exception as e:
        return f"ERROR: {e}"


def normalize_mutation(mutation):
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    match = re.match(r"([ATGC])\\[([ATGC]>[ATGC])\\]([ATGC])", mutation)
    if match:
        ref, change, alt = match.groups()
        ref_from, ref_to = change.split('>')
        if ref_from not in ['C', 'T']:
            ref_from = complement[ref_from]
            ref_to = complement[ref_to]
            ref = complement[alt]
            alt = complement[ref]
        return f"{ref}[{ref_from}>{ref_to}]{alt}"
    return mutation


def assign_genomic_bin(df):
    df['Genomic_Bin'] = df['Chromosome'].astype(str) + '_' + (((df['Start'] - 1) // 1_000_000) * 1_000_000).astype(str)
    return df
