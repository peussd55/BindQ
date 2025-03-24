from Bio import SeqIO, PDB
import os

def pdb_to_fasta(pdb_file, output_fasta):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    pdb_id = os.path.basename(pdb_file).split('.')[0]
    
    # Combined sequence from all chains
    combined_seq = ""
    
    for model in structure:
        for chain in model:
            # Get sequence for this specific chain
            chain_seq = "".join([PDB.Polypeptide.three_to_one(res.get_resname()) 
                          for res in chain.get_residues() 
                          if res.get_resname() in PDB.Polypeptide.protein_letters_3to1])
            
            # Add to combined sequence
            combined_seq += chain_seq
    
    # Create single FASTA record with combined sequence
    fasta_record = f">{pdb_id}\n{combined_seq}"
    
    with open(output_fasta, "w") as f:
        f.write(fasta_record)
        
pdb_to_fasta('./input_pdb/5O2T.pdb', 'test.fasta')