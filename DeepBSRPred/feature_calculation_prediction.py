import glob
import os
import pandas as pd
import re
import requests
import argparse
import tempfile
import shutil
import subprocess

from Bio import SeqIO, PDB
from Bio.PDB import *
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.ResidueDepth import residue_depth

import tensorflow as tf
import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
import math
import warnings

warnings.filterwarnings('ignore')

aa = 'ARNDCQEGHILKMFPSTWYV'
aali = list(aa)

# AA groups
polar = ['N','Q', 'S', 'T', 'P']
aromatic = ['Y', 'F', 'W']
pos_charge = ['K', 'R', 'H']
Neg_charge = ['D', 'E']
Sul_containing = ['C','M']
Aliphatic = ['G','A', 'L','I', 'V']

# read numerical values of amino acids from files
df1 = pd.read_csv('./data/49_properties_numerical_Values.csv')
prop_list = [line.strip() for line in open('./data/prop_49_list.csv').readlines()[1:]]

def GetQueryfromClustalAlignment(clustalfile):
    with open(clustalfile, 'r') as f:
        for line in f:
            if line.startswith('QuerySeq'):
                return line.replace("\n","").split()[-1]
    return ""

# ASA of amino acids
dict_asa = {'A':110.2,'D':144.1,'C':140.4,'E':174.7,
          'F':200.7,'G':78.7,'H':181.9,'I':185,'L':183.1,
          'M':200.1,'N':146.4,'P':141.9,'Q':178.6,'R':229,
          'S':117.2,'T':138.7,'V':153.7,'W':240.5,'Y':213.7,'K':205.7,'X':0,'U':0}

def pdb_to_fasta(pdb_file, output_fasta):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    pdb_id = os.path.basename(pdb_file).split('.')[0]

    seq_records = []
    for model in structure:
        for chain in model:
            seq = "".join([PDB.Polypeptide.three_to_one(res.get_resname())
                           for res in chain.get_residues() if res.get_resname() in PDB.Polypeptide.protein_letters_3to1])

            fasta_header = f">{pdb_id}_{chain.id}"
            seq_records.append(f"{fasta_header}\n{seq}")

    with open(output_fasta, "w") as f:
        f.write("\n".join(seq_records))


parser = argparse.ArgumentParser(description="Feature Calculation for Protein Sequences and Structures")
parser.add_argument("--input_file", type=str, required=True, help="Path to input file (.fasta or .pdb)")
args = parser.parse_args()

input_file = args.input_file
file_ext = os.path.splitext(input_file)[1].lower()
id1 = os.path.basename(input_file).split('.')[0]

output_pdb_dir = "./AlphaFold_out/"
os.makedirs(output_pdb_dir, exist_ok=True)

if file_ext == ".pdb":
    pdb_output_path = f"{output_pdb_dir}/{id1}.pdb"
    if not os.path.exists(pdb_output_path):
        shutil.copy(input_file, pdb_output_path)

    fasta_output = f'./input_fasta/{id1}.fasta'
    if not os.path.exists(fasta_output):
        pdb_to_fasta(input_file, fasta_output)

    input_file = fasta_output


print(f"Processing sequence-based features for {id1}...")

cmd = f"psiblast -query {input_file} -db ./blastdb/database_90.fasta -out ./psi_blast_out/{id1}.txt -num_iterations 3 -out_ascii_pssm ./pssm_out/{id1}.pssm"
os.system(cmd)

with open(f'./pssm_out/{id1}.pssm','r') as f_in, open(f'./pssm_parsed/{id1}.csv','w') as f_out:
    f_out.write('RES,' + ','.join(aali) + '\n')
    for line in f_in.readlines()[3:-6]:  # Adjusted for PSSM file format
        f_out.write(','.join(line.split()[1:22]) + '\n')

seq = open(f'{input_file}','r').readlines()[1].strip()
with open(f'./window_11/{id1}.csv','w') as f_out:
    f_out.write('p,aro,pos_c,neg_c,sul_c,ali\n')
    for i in range(len(seq)):
        p, aro, pos_c, neg_c, sul_c, ali = 0, 0, 0, 0, 0, 0
        window_start = max(0, i - 5)  # Corrected window start
        window_end = min(len(seq), i + 6)  # Corrected window end
        window_5 = seq[window_start:window_end]

        for item in window_5:
            if item in polar:   p += 1
            if item in aromatic: aro += 1
            if item in pos_charge: pos_c += 1
            if item in Neg_charge: neg_c += 1
            if item in Sul_containing: sul_c += 1
            if item in Aliphatic: ali += 1
        f_out.write(f"{p},{aro},{pos_c},{neg_c},{sul_c},{ali}\n")

f_out = pd.DataFrame()
for i in range(len(seq)):
    window_start = max(0, i - 5)
    window_end = min(len(seq), i + 6)
    window = seq[window_start:window_end]
    li = list(window)
    temp = pd.DataFrame()
    temp['sum'] = df1[li].sum(axis=1)
    f_out = pd.concat([f_out, temp.T])

f_out.set_axis(prop_list, axis=1, inplace=True)
f_out.to_csv('./AA_index/' + id1 + '.csv', index=False)

cmd = f"blastp -query ./input_fasta/{id1}.fasta -db ./blastdb/database_90.fasta -num_threads 5 -outfmt 4 -out ./out_/{id1}.out -max_target_seqs 100"
os.system(cmd)

h = open(f"./input_fasta/{id1}.fasta").readlines()
g = open(f"./out_/{id1}.out").readlines()
p = re.compile(r'[OPQ][A-Z0-9]{5} ')
IDs = []
for line in g:
    if (match := p.match(line)) is not None:  # Use walrus operator
        ID = match.group().rstrip()
        if ID not in IDs:
            IDs.append(ID)

seq = h[1].strip()

with open(f"./msa_/{id1}_all.fasta", 'w') as q:
    q.write(f">QuerySeq\n{seq}\n\n")
    base_url = 'https://rest.uniprot.org/uniprotkb/search?query'
    accessions = ")OR(accession:".join(IDs)
    response = requests.get(f"{base_url}=(accession:{accessions})&format=fasta")
    q.write(response.text)

    records = SeqIO.parse(f"./msa_/{id1}_all.fasta", 'fasta')
    unusual = re.compile(r'[BJOUXZ()]')
    updates = [record for record in records if not unusual.search(str(record.seq))]
    SeqIO.write(updates, f"./msa_/{id1}_all.fasta", "fasta")
    count_blast = len(updates)

os.system(f"./mafft-linux64/mafft.bat --clustalout ./msa_/{id1}_all.fasta > ./msa_/{id1}_all.msa")
os.system(f"java -jar ./compbio-conservation-1.1.jar -i=./msa_/{id1}_all.msa -o=./out_/{id1}.AAcons -n -f=RESULT_WITH_NO_ALIGNMENT")

fi = GetQueryfromClustalAlignment(f"./msa_/{id1}_all.msa")
letters = [f"{seq[i-1]}{i}" for i in range(1, len(seq) + 1) if seq[i-1] != '-']
scores = {x: [] for x in letters}

with open(f"./out_/{id1}.AAcons") as aa:
    for line in aa:
        if line.strip() and not line.isspace():
            line_parts = line.strip().split()
            try:
                sc = list(map(float, line_parts[1:]))
                tmpsc = [val for i, val in enumerate(sc) if fi[i] != '-']
                for i, val in enumerate(tmpsc):
                    scores[letters[i]].append(val)
            except (ValueError, IndexError):  # Handle potential errors
                pass

with open(f"./AAcon_out/{id1}.csv", "w") as out:
    out.write("AA,KABAT,JORES,SCHNEIDER,SHENKIN,GERSTEIN,TAYLOR_GAPS,TAYLOR_NO_GAPS,VELIBIL,KARLIN,ARMON,THOMPSON,NOT_LANCET,MIRNY,WILLIAMSON,LANDGRAF,SANDER,VALDAR,SMERFS\n")
    for u, v in scores.items():
        out.write(f"{u},{','.join(map(str, v))}\n")

df3 = pd.read_csv(f'./pssm_parsed/{id1}.csv')
df2 = pd.DataFrame()
cols = df3.columns[1:-2]
df1 = df3[cols]

for i in range(len(df1)):
    indices = [i - 2, i - 1, i, i + 1, i + 2]
    valid_indices = [idx for idx in indices if 0 <= idx < len(df1)]
    x = sum(df1.iloc[idx] for idx in valid_indices)
    df2 = df2.append(x, ignore_index=True)

df2['RES'] = df3['RES']
df2.to_csv(f'./pssm_sw_5/{id1}.csv', index=False)

try:
    if not os.path.exists(f"./AlphaFold_out/{id1}.pdb"):
        if len(id1) >= 6:
            url = f"https://alphafold.ebi.ac.uk/files/AF-{id1}-F1-model_v4.pdb"
        else:
            url = f"https://files.rcsb.org/download/{id1}.pdb"
        cmd = f"wget {url} -O ./AlphaFold_out/{id1}.pdb"
        os.system(cmd)
    else:
        print(f"./AlphaFold_out/{id1}.pdb already exists. Skipping download.")

    dssp_cmd = f'./dssp/build/mkdssp ./AlphaFold_out/{id1}.pdb ./dssp_out/{id1}.dssp'
    os.system(dssp_cmd)

    try:
        with open(f'./dssp_out/{id1}.dssp', 'r') as f1, open(f'./dssp_csv/{id1}.csv', 'w') as f_out:
            f_out.write('res,pos,ASA,SEC_STR\n')
            for line in f1.readlines()[28:]:  # Corrected loop
                try:
                    if line[13:14] in ('X', 'U'):
                        continue
                    asa = float(line[35:39].strip())
                    pos = int(line[6:11])
                    res = line[13:14]

                    sec_str_code = line[16:17].strip()
                    if sec_str_code in ('H', 'G', 'I'):
                        sec_str = 'Helix'
                    elif sec_str_code in ('E', 'B'):
                        sec_str = "Sheet"
                    else:
                        sec_str = 'Coil'
                    f_out.write(f'{res},{pos},{asa},{sec_str}\n')
                except ValueError:
                    pass
    except FileNotFoundError:
        print(f'check dssp file: {id1}')
        with open(f'./dssp_csv/{id1}.csv', 'w') as f_out:
            f_out.write('res,pos,ASA,SEC_STR\n')

    try:
        df11 = pd.read_csv(f'./dssp_csv/{id1}.csv')
        with open(f'./input_fasta/{id1}.fasta', 'r') as f1:
            df11['ASA_PER'] = ''
            for i in range(len(df11)):
                try:
                    df11.loc[i, 'ASA_PER'] = round((float(df11['ASA'][i] * 100) / float(dict_asa[df11['res'][i]])), 2)
                except (KeyError, TypeError, ValueError):
                    df11.loc[i, 'ASA_PER'] = 0.0  # Or some other default
            sequences = []
            current_seq = []
            for line in f1:
                if line.startswith(">"):
                    if current_seq:
                        sequences.append("".join(current_seq))
                        current_seq = []
                else:
                    current_seq.append(line.strip())
            if current_seq:
                sequences.append("".join(current_seq))

            try:
                df11['seq'] = "".join(sequences)[0:len(df11)]
            except:
                print("Sequence length mismatch", id1)
            df11.to_csv(f'./ASA_PER_protein/{id1}.csv', index=False)
    except FileNotFoundError:
        print(f"dssp_csv not found: {id1}")

    parser = PDBParser()

    try:
        df11 = pd.read_csv(f'./ASA_PER_protein/{id1}.csv')
        df11['residue_depth'] = ''
        structure = parser.get_structure(id1, f"./AlphaFold_out/{id1}.pdb")
        model = structure[0]
        surface = get_surface(model)
        chain1 = [chain.id for chain in structure.get_chains()]  # Use list comprehension
        chain = model[chain1[0]]
        rd = ResidueDepth(model)
        RD = []

        # Iterate through residues with error handling
        for residue in chain:
            try:
                res_id = (chain1[0], residue.id)  # Create the tuple correctly
                depth = rd[res_id][0]
                RD.append(round(depth, 2))

            except KeyError:
                RD.append(0.0)  # Default value for missing residues
            except Exception as e:  # Catch other potential errors
                print(f"Error processing residue {residue.id}: {e}")
                RD.append(0.0)
        df11['residue_depth'] = RD[0:len(df11)]  # Ensure correct length
        df11.to_csv(f'./ASA_PER_residue_depth/{id1}.csv', index=False)


    except (FileNotFoundError, KeyError, Exception) as e:
        print(f"Error calculating residue depth for {id1}: {e}.  Skipping.")
        try:
            df11 = pd.read_csv(f'./ASA_PER_protein/{id1}.csv')
        except FileNotFoundError:
            df11 = pd.DataFrame() # Create empty if ASA_PER file also missing
        df11['residue_depth'] = [0] * len(df11)
        df11.to_csv(f'./ASA_PER_residue_depth/{id1}.csv', index=False)


    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
         'ALA': 'A', 'VAL': 'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}

    try:
        with open(f'./AlphaFold_out/{id1}.pdb') as f1:
            df11 = pd.read_csv(f'./ASA_PER_residue_depth/{id1}.csv')
            pdb_lines = f1.readlines()  # Read all lines into memory

        list1 = []
        for i in range(len(pdb_lines)):
            if pdb_lines[i].startswith('ATOM'):
                count = 0
                try:  # Add error handling within the loop
                    residue = pdb_lines[i][17:20].strip()  # Use string slicing
                    atom = pdb_lines[i][12:16].strip()
                    if atom == 'CA' and residue in d:
                        residue_pos = int(pdb_lines[i][22:26].strip())  # Convert to int
                        x = float(pdb_lines[i][30:38].strip())
                        y = float(pdb_lines[i][38:46].strip())
                        z = float(pdb_lines[i][46:54].strip())

                        for j in range(len(pdb_lines)):
                            if pdb_lines[j].startswith('ATOM'):
                                residue1 = pdb_lines[j][17:20].strip()
                                atom1 = pdb_lines[j][12:16].strip()
                                if atom1 == 'CA' and residue1 in d:
                                    residue_pos1 = int(pdb_lines[j][22:26].strip())
                                    x1 = float(pdb_lines[j][30:38].strip())
                                    y1 = float(pdb_lines[j][38:46].strip())
                                    z1 = float(pdb_lines[j][46:54].strip())
                                    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2 + (z1 - z)**2)
                                    if distance <= 8:
                                        count += 1
                except (ValueError, IndexError) as e: # Handle string parsing errors
                    print(f"Error parsing PDB line {i}: {e}")
                    continue  # Skip to the next line

                list1.append(count)

        if len(list1) > len(df11):
            df11['contact_count'] = list1[0:len(df11)]
        elif len(list1) < len(df11):
             df11['contact_count'] = list1 + [0] * (len(df11) - len(list1))  # Pad with zeros
        else:
            df11['contact_count'] = list1
        df11.to_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv', index=False)

    except FileNotFoundError:
        print(f"PDB or ASA/Residue Depth file not found: {id1}")
        try: #Try to load previous, and add defaults.
            df11 = pd.read_csv(f'./ASA_PER_residue_depth/{id1}.csv')
        except:
            df11 = pd.DataFrame()
        df11['contact_count'] = [0] * len(df11)
        df11.to_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv', index=False)

    print("here hbplus")
    print("="*100)
    os.chdir('./hbplus_out/')
    cmd = f'hbplus -P ../AlphaFold_out/{id1}.pdb'
    os.system(cmd)
    os.chdir('../')
    print("finish hbplus")
    print("="*100)

    try:
        with open(f"./hbplus_out/{id1}.hb2") as h:
            df = pd.read_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv')
            df['hb_donor'] = [0] * len(df)
            df['hb_acceptor'] = [0] * len(df)
            hb_lines = h.readlines()[8:]

        for i in range(len(df)):
            res = df['res'][i]
            pos = df['pos'][i]
            for k in hb_lines:
                try:
                    k = k.rstrip()
                    donor_res = k[:13].split()[0][-3:]
                    donor_pos = int(k[:13].split()[0][1:5])
                    acceptor_res = k[14:27].split()[0][-3:]
                    acceptor_pos = int(k[14:27].split()[0][1:5])

                    if d.get(donor_res) == res and donor_pos == pos:
                        df.loc[i, 'hb_donor'] += 1
                    if d.get(acceptor_res) == res and acceptor_pos == pos:
                        df.loc[i, 'hb_acceptor'] += 1
                except (KeyError, IndexError, ValueError):
                    pass
    except FileNotFoundError:
        print(f'hbplus file or contact file not found: {id1}')
        try:
            df = pd.read_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv')
        except:
            df = pd.DataFrame() # Create empty dataframe.
        df['hb_donor'] = [0] * len(df)
        df['hb_acceptor'] = [0] * len(df)

    df.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor/{id1}.csv', index=False)

    df.drop(['pos', 'res'], axis='columns', inplace=True, errors='ignore')
    df.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/{id1}.csv', index=False)


except KeyError:
    print(f"KeyError during main processing for {id1}")
    # Removed Naccess related columns (already done in the previous steps)
    colss = ['ASA', 'ASA_PER', 'residue_depth', 'contact_count', 'hb_donor', 'hb_acceptor']
    try:
        # Attempt to read the pssm_sw_5 file to get sequence
        test_df1 = pd.read_csv(f'./pssm_sw_5/{id1}.csv')
        seq_series = test_df1['RES']
    except FileNotFoundError:
        print(f"Could not find PSSM file, attempting to use input FASTA for {id1}")
        try:
            with open(f'./input_fasta/{id1}.fasta') as f:
                lines = f.readlines()
            seq = lines[1].strip()
            seq_series = pd.Series(list(seq))  # Create a Series for easier handling
        except FileNotFoundError:
            print(f"Could not find input FASTA file for {id1}.  Cannot proceed.")
            exit()

    # Create a DataFrame with default values
    a = pd.DataFrame(0, index=range(len(seq_series)), columns=colss)
    a.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/{id1}.csv', index=False)


# Merge all data
cols1 = ['RES', 'GEIM800105', 'F', 'D', 'ISOY800102', 'LAWE840101', 'aN', 'pK', 'Ht', 'p', 'mc_real',
         'ali', 'Pf-s', 's', 'MIYS850101', 'V0', 'hb_donor', 'Ra', 'S', 'LIFS790101', ' THOMPSON', 'G',
         'Non_polar_abs', 'FASG760101', 'A', 'C', 'E', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'W',
         'pssm_sum', 'aro', 'pos_c', 'neg_c', 'sul_c', 'K0', 'Hp', 'pHi', 'Mw', 'Bl']

df = pd.read_csv(f'./window_11/{id1}.csv')
df1 = pd.read_csv(f'./pssm_sw_5/{id1}.csv')
a = pd.read_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/{id1}.csv')

aa_index = pd.read_csv(f'./AA_index/{id1}.csv')
df_cons = pd.read_csv(f'./AAcon_out/{id1}.csv')
a['ASA_PER_AVG'] = ''

for i in range(len(a)):
    indices = [i - 2, i - 1, i, i + 1, i + 2]
    valid_indices = [idx for idx in indices if 0 <= idx < len(a) and 'ASA_PER' in a.columns] # Check column exists
    if valid_indices:
        a.loc[i, 'ASA_PER_AVG'] = a['ASA_PER'].iloc[valid_indices].mean()
    else:
        a.loc[i, 'ASA_PER_AVG'] = 0.0  # Default if no valid indices

a['seq'] = df1['RES']
df1['pssm_sum'] = df1.iloc[:, :-1].sum(axis=1, skipna=True)  # Use iloc for numerical indexing


# Concatenate DataFrames
df3 = pd.concat([df1, df, aa_index, a, df_cons], axis=1)

# Handle potential missing columns and set default values
df3['V'] = df3['V0'] if 'V0' in df3.columns else 0.0
df3['Y'] = df3['p'] if 'p' in df3.columns else 0.0


# --- CRUCIAL FIX ---
# 1. Check if 'P' exists.
if 'P' in df3.columns:
    # 2. Ensure df3['P'] is a Series.  If it's a DataFrame, take the first column.
    p_series = df3['P'].copy()  # Make a copy to avoid SettingWithCopyWarning
    if isinstance(p_series, pd.DataFrame):
        p_series = p_series.iloc[:, 0]  # Take the first column

    # 3.  Check length and handle mismatches
    if len(p_series) == len(df3):
          df3['P.1'] = p_series # Assign
    elif len(p_series) < len(df3):  # Pad if shorter
          df3['P.1'] = list(p_series) + [0.0] * (len(df3) - len(p_series))
    else: # Truncate if longer
        df3['P.1'] = p_series[:len(df3)]

else:  # 'P' does not exist at all, create 'P.1' filled with default.
    df3['P.1'] = 0.0
# --- END CRUCIAL FIX ---


# Removed Naccess related columns from cols11
cols11 = ['RES','GEIM800105', 'F', 'D', 'ISOY800102', 'LAWE840101', 'aN', 'pK', 'Ht', 'p', 'mc_real', 'V', 'ali', 'Pf-s', 's', 'MIYS850101', 'V0', 'hb_donor', 'Ra', 'S', 'LIFS790101', ' THOMPSON', 'G', 'Non_polar_abs', 'FASG760101', 'A', 'C', 'E', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'W', 'Y', 'pssm_sum', 'aro', 'pos_c', 'neg_c', 'sul_c', 'K0','Hp', 'P.1', 'pHi', 'Mw', 'Bl']  # Added 'P.1'


weighted_model = tf.keras.models.load_model(f'./saved_model/EC_model.h5', compile=False)
scaler = StandardScaler()

with open(f'./input_fasta/{id1}.fasta', 'r') as f:
    lines = f.readlines()

dict_temp = {}
for i in range(0, len(lines) - 1, 2):
    dict_temp[lines[i].split('>')[1].strip()] = lines[i + 1].strip()

available = open('./list.txt', 'r').read().splitlines()
p_threshold = 0.49

for key, seq_val in dict_temp.items():
    data_df1 = pd.DataFrame()
    if key in available:
        df = pd.read_csv(f'./datasets/merged/{key}.csv')
        # --- FIX: Drop non-numeric columns before scaling ---
        df_numeric = df[cols1].select_dtypes(include=np.number)
        train_features = scaler.fit_transform(df_numeric)
        # --- END FIX ---
        data_df1['Residue'] = df['RES']
        train_predictions_weighted = weighted_model.predict(train_features, batch_size=20)
        prediction_label_train = [int(p >= p_threshold) for p in train_predictions_weighted]
        data_df1['Prediction'] = prediction_label_train
        data_df1.T.to_csv(f'./results/{id1}_result.csv')
        print(f"Results are saved in results directory with file name as:{id1}_result.csv file")

    else:
        df3 = df3.loc[:, ~df3.columns.duplicated()].copy()
        # Handle missing columns before scaling:
        for col in cols11:
            if col not in df3.columns:
                df3[col] = 0.0  # Or another appropriate default

        # --- FIX: Drop non-numeric columns before scaling ---
        df3_numeric = df3[cols11].select_dtypes(include=np.number)
        train_features = scaler.fit_transform(df3_numeric)
        # --- END FIX ---
        # Get the 'RES' column *before* any potential errors or modifications
        try:
          residue_col = df3['RES'].copy()
        except KeyError:
          print("RES column not present. Creating from sequence length.")
          residue_col = pd.Series(list(seq_val))
        data_df1['Residue'] = residue_col
        train_predictions_weighted = weighted_model.predict(train_features, batch_size=20)
        prediction_label_train = [int(p >= p_threshold) for p in train_predictions_weighted]
        data_df1['Prediction'] = prediction_label_train
        data_df1.T.to_csv(f'./results/{id1}_result.csv')
        print(f"Results are saved in the results directory with file name: {id1}_result.csv")