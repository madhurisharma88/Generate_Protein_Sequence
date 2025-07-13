from Bio.Blast import NCBIWWW
from Bio.Blast import NCBIXML

# Example protein sequence (hemoglobin beta)
sequence = "MVHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQR"

# Run BLASTP (protein-protein BLAST) 
print("Running BLAST...")
result_handle = NCBIWWW.qblast("blastp", "nr", sequence)

# Parse the BLAST result
blast_records = NCBIXML.read(result_handle)

# Print top 5 matches
print("\nTop 5 similar proteins:")
for alignment in blast_records.alignments[:5]:
    print(f"\n> {alignment.title}")
    for hsp in alignment.hsps:
        print(f"  Identity: {hsp.identities}/{hsp.align_length}")
        print(f"  E-value: {hsp.expect}")
        break  # Only show the first HSP (alignment block)
