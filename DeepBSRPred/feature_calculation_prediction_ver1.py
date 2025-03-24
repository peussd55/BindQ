#!/usr/bin/env python
# coding: utf-8
##import all package
import glob
import multiprocessing as mp
import os
import pandas as pd
import re
import requests
import freesasa


from Bio import SeqIO, PDB
from Bio.PDB import *
from Bio.PDB.ResidueDepth import ResidueDepth
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.ResidueDepth import get_surface
from Bio.PDB.ResidueDepth import residue_depth
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.AbstractPropertyMap import (
    AbstractResiduePropertyMap,
    AbstractAtomPropertyMap,
)
import argparse
import random
import tempfile
import shutil
import subprocess
import tensorflow as tf
import keras
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib
from matplotlib import pyplot
# from scipy import interp
from sklearn.preprocessing import StandardScaler,binarize
from keras import Sequential
from keras.layers import LSTM
from keras.layers import Dense, Activation, Dropout
import math

### To calculate sequence-based features
### Change blastDB name and give path
### pass FASTA file
import warnings
warnings.filterwarnings('ignore')

aa = 'ARNDCQEGHILKMFPSTWYV'
aali = []
for a in aa:
    aali.append(a)

### AA groups
polar = ['N','Q', 'S', 'T', 'P']
aromatic = ['Y', 'F', 'W']
pos_charge = ['K', 'R', 'H']
Neg_charge = ['D', 'E']
Sul_containing = ['C','M']
Aliphatic = ['G','A', 'L','I', 'V']

### read numerical values of amino acids from files
df1 = pd.read_csv('./data/49_properties_numerical_Values.csv')
prop_ = open('./data/prop_49_list.csv').readlines()[1:]
prop_list = []
for j in prop_:
    prop_list.append(j.strip())
### run query to fetch clustalfile
def GetQueryfromClustalAlignment( clustalfile ):
    seq = ''
    with open( clustalfile, 'r' ) as f:
        for line in f:
            if line.startswith( 'QuerySeq' ):
                seq += line.replace("\n","").split()[-1]
    return seq

### ASA of amino acids
dict_asa = {'A':110.2,'D':144.1,'C':140.4,'E':174.7,
          'F':200.7,'G':78.7,'H':181.9,'I':185,'L':183.1,
          'M':200.1,'N':146.4,'P':141.9,'Q':178.6,'R':229,
          'S':117.2,'T':138.7,'V':153.7,'W':240.5,'Y':213.7,'K':205.7,'X':0,'U':0}

### NACCESS CODES STARTS HERE
# def run_naccess(model, pdb_file, probe_size=None, z_slice=None, naccess="./Naccess/naccess", temp_path="./tmp/"): ###change path here for naccess
#     """Run naccess for a pdb file."""
#     # make temp directory;
#     tmp_path = tempfile.mkdtemp(dir=temp_path)

#     # file name must end with '.pdb' to work with NACCESS
#     # -> create temp file of existing pdb
#     #    or write model to temp file
#     handle, tmp_pdb_file = tempfile.mkstemp(".pdb", dir=tmp_path)
#     os.close(handle)
#     if pdb_file:
#         pdb_file = os.path.abspath(pdb_file)
#         shutil.copy(pdb_file, tmp_pdb_file)
#     else:
#         writer = PDBIO()
#         writer.set_structure(model.get_parent())
#         writer.save(tmp_pdb_file)

#     # chdir to temp directory, as NACCESS writes to current working directory
#     old_dir = os.getcwd()
#     os.chdir(tmp_path)

#     # create the command line and run
#     # catch standard out & err
#     command = [naccess, tmp_pdb_file]
#     if probe_size:
#         command.extend(["-p", probe_size])
#     if z_slice:
#         command.extend(["-z", z_slice])

#     p = subprocess.Popen(
#         command, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#     )
#     out, err = p.communicate()
#     os.chdir(old_dir)

#     rsa_file = tmp_pdb_file[:-4] + ".rsa"
#     asa_file = tmp_pdb_file[:-4] + ".asa"
#     # Alert user for errors
#     if err.strip():
#         warnings.warn(err)

#     if (not os.path.exists(rsa_file)) or (not os.path.exists(asa_file)):
#         raise Exception("NACCESS did not execute or finish properly.")

#     # get the output, then delete the temp directory
#     #f_out_rsa = open('../Naccess_out/'+id1+'.rsa','w')
#     #f_out_asa = open('../Naccess_out/'+id1+'.asa','w')
#     with open(rsa_file) as rf:
#         rsa_data = rf.readlines()
#         #f_out_rsa.writelines(rf)
#     with open(asa_file) as af:
#         asa_data = af.readlines()
#        # f_out_asa.writelines(af)
#    # f_out_rsa.close()
#    # f_out_asa.close()
#     # shutil.rmtree(tmp_path, ignore_errors=True)
#     return rsa_data, asa_data



def run_freesasa(model, pdb_file, probe_size=1.4, temp_path="./tmp/"):
    """Run freesasa for a pdb file and return data in a format similar to NACCESS."""
    # 임시 디렉토리 생성
    tmp_path = tempfile.mkdtemp(dir=temp_path)
    
    # 임시 파일 생성
    handle, tmp_pdb_file = tempfile.mkstemp(".pdb", dir=tmp_path)
    os.close(handle)
    # PDB 파일 준비
    if pdb_file:
        pdb_file = os.path.abspath(pdb_file)
        shutil.copy(pdb_file, tmp_pdb_file)
    else:
        writer = PDBIO()
        writer.set_structure(model.get_parent())
        writer.save(tmp_pdb_file)
        
    command = ["freesasa", "--format=rsa", f"{tmp_pdb_file}"]
    result = subprocess.run(command, capture_output=True, text=True)
    # 결과를 파일로 내보내기 (RSA 형식으로)
    rsa_file = os.path.join(tmp_path, f"{tmp_pdb_file[-7:-4]}.rsa")

    # 자체 형식 출력 파일 생성
    with open(rsa_file, 'w') as f:
        f.write(result.stdout)
    with open("./Naccess/output.rsa", 'w') as f:
        f.write(result.stdout)
    
    # 출력 결과를 변수에 저장
    with open(rsa_file) as rf:
        rsa_data = rf.readlines()
    
    return rsa_data


def process_rsa_data(rsa_data):
    """Process the .rsa output file: residue level SASA data."""
    naccess_rel_dict = {}
    for line in rsa_data:
        if line.startswith("RES"):
            res_name = line[4:7]
            chain_id = line[8]
            resseq = int(line[9:13])
            icode = line[13]
            res_id = (" ", resseq, icode)
            naccess_rel_dict[(chain_id, res_id)] = {
                "res_name": res_name,
                "all_atoms_abs": float(line[16:22]),
                "all_atoms_rel": float(line[23:28]),
                "side_chain_abs": float(line[29:35]),
                "side_chain_rel": float(line[36:41]),
                "main_chain_abs": float(line[42:48]),
                "main_chain_rel": float(line[49:54]),
                "non_polar_abs": float(line[55:61]),
                "non_polar_rel": float(line[62:67]),
                "all_polar_abs": float(line[68:74]),
                "all_polar_rel": float(line[75:80]),
            }
    return naccess_rel_dict


def process_asa_data(rsa_data):
    """Process the .asa output file: atomic level SASA data."""
    naccess_atom_dict = {}
    for line in rsa_data:
        full_atom_id = line[12:16]
        atom_id = full_atom_id.strip()
        chainid = line[21]
        resseq = int(line[22:26])
        icode = line[26]
        res_id = (" ", resseq, icode)
        id = (chainid, res_id, atom_id)
        asa = line[54:62]  # solvent accessibility in Angstrom^2
        naccess_atom_dict[id] = asa
    return naccess_atom_dict
## NACCESS CODES ENDS HERE

# PDB 파일을 FASTA로 변환하는 함수
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
# fasta_list  = glob.glob('./input_fasta/*.fasta') #### keep input files in input_fasta folder and pass path here
# pdb_list  = glob.glob('./input_pdb/*.pdb')

# # 1️⃣ **PDB 직접 입력 시 → FASTA 변환 후 리스트에 추가**
# for pdb_file in pdb_list:
#     id1 = os.path.basename(pdb_file)

# **✅ Argument Parser 추가**
parser = argparse.ArgumentParser(description="Feature Calculation for Protein Sequences and Structures")
parser.add_argument("--input_file", type=str, required=True, help="Path to input file (.fasta or .pdb)")
args = parser.parse_args()

# **✅ 입력 파일 검사 및 처리**
input_file = args.input_file
file_ext = os.path.splitext(input_file)[1].lower()  # 확장자 확인
id1 = os.path.basename(input_file).split('.')[0]  # 파일명에서 ID 추출

# PDB 저장 디렉터리 설정
output_pdb_dir = "./AlphaFold_out/"
os.makedirs(output_pdb_dir, exist_ok=True)

# ✅ **PDB 파일 입력 시 처리**
if file_ext == ".pdb":
    pdb_output_path = f"{output_pdb_dir}/{id1}.pdb"
    
    # 기존 코드와 호환성을 위해 AlphaFold_out 폴더로 PDB 복사
    if not os.path.exists(pdb_output_path):
        shutil.copy(input_file, pdb_output_path)

    # PDB → FASTA 변환
    fasta_output = f'./input_fasta/{id1}.fasta'
    if not os.path.exists(fasta_output):
        pdb_to_fasta(input_file, fasta_output)
    print('complete pdb to fasta')
    input_file = fasta_output  # 이후 처리 로직을 위해 FASTA 파일로 변경

### main code is here
# if file_ext == ".fasta":
### id1 is the name of file
# SEQUENCE BASED ##################
print(f"Processing sequence-based features for {id1}...")

cmd = f"psiblast -query {input_file} -db ./blastdb/database_90.fasta -out ./psi_blast_out/{id1}.txt -num_iterations 3 -out_ascii_pssm ./pssm_out/{id1}.pssm"  # blastdb
os.system(cmd)
## read pssm and convert to csv with header for input files
f1 = open(f'./pssm_out/{id1}.pssm','r').readlines()[2:-6]
f_out = open(f'./pssm_parsed/{id1}.csv','w')
f_out.writelines('RES'+','+','.join(aali)+'\n')
for i in f1[1:]:
    f_out.writelines(','.join(i.split()[1:22])+'\n')
f_out.close()
# SET  SLIDING WINDOW
win = 11
seq = open(f'{input_file}','r').readlines()[1].strip()  # read sequence
f_out = open(f'./window_11/{id1}.csv','w')  # open files
f_out.write('p'+","+'aro'+","+'pos_c'+","+'neg_c'+","+'sul_c'+","+'ali'+"\n")
## take window of 11 and calculate properties
for i in range(0,len(seq)):
    p = 0
    aro = 0
    pos_c = 0
    neg_c = 0
    sul_c = 0
    ali = 0
    if i-4 < 0:
        window_5 = seq[:i+3]
    elif i+5 >len(seq):
        window_5 = seq[i-2:]
    else:
        window_5 = seq[i-4:i+5]

    for item in window_5:
        if item in polar:
            p += 1
        if item in aromatic:
            aro += 1
        if item in pos_charge:
            pos_c +=1
        if item in Neg_charge:
            neg_c += 1
        if item in Sul_containing:
            sul_c += 1
        if item in Aliphatic:
            ali += 1
    f_out.write(str(p)+","+str(aro)+","+str(pos_c)+","+str(neg_c)+","+str(sul_c)+","+str(ali)+"\n")
f_out.close()

# AAINDEX BASED PROPERTIES
f_out = pd.DataFrame()
for i in range(0,len(seq)):
    temp = pd.DataFrame()
    if i-4 < 0:
        window = seq[:i+4]
    elif i+5 >len(seq):
        window = seq[i-2:]
    else:
        window = seq[i-4:i+5]
    li = list(window)
    df1 = pd.read_csv('./data/49_properties_numerical_Values.csv')
    temp['sum'] = df1[li].sum(axis=1)
    f_out = f_out.append(temp.T)
f_out.set_axis(prop_list,axis = 1,inplace=True)
f_out.to_csv('./AA_index/'+id1+'.csv',index = False)
# AACon
cmd = "blastp -query ./input_fasta/{0}.fasta -db ./blastdb/database_90.fasta -num_threads 5 -outfmt 4 -out ./out_/{0}.out -max_target_seqs 100".format(id1)
os.system(cmd)
#print (i)
h=open("./input_fasta/{}.fasta".format(id1)).readlines()
g=open("./out_/{}.out".format(id1)).readlines()
p = re.compile('[OPQ]{1}[A-Z0-9]{5} ')
IDs=[]
for line in g:
    if re.match( p, line ) != None:
        ID = re.match( p, line ).group().rstrip()
        if ID not in IDs:
            IDs.append(ID)
#print(IDs)
#seq=''
#for j in h:
    # j=j.rstrip()
    # if j[0]!=">":
    #    seq+=j
seq = h[1].strip()
#print (seq)

with open("./msa_/{}_all.fasta".format(id1), 'w' ) as q:
    q.write( ">QuerySeq\n{}\n\n".format(seq) )
    #base_url = 'https://www.uniprot.org/uniprot/?query'
    base_url = 'https://rest.uniprot.org/uniprotkb/search?query'
    response = requests.get( "%s=%s&format=fasta"%(base_url,"(accession:" + ")OR(accession:".join(IDs) + ")" ) )
    q.write( response.text )

    records = SeqIO.parse( "./msa_/{}_all.fasta".format(id1), 'fasta' )
    unusual = re.compile( '[BJOUXZ()]' )
    updates = []
    count   = 0
    for record in records:
        if re.search(unusual, str(record.seq) ) == None:
            updates.append( record )
            count += 1
        else:
            #print( "\t" + record.id )
            next
    with open( "./msa_/{}_all.fasta".format(id1), 'w' ) as output_handle:
        SeqIO.write(updates, output_handle, "fasta")
    count_blast = count


os.system("./mafft-linux64/mafft.bat --clustalout ./msa_/{0}_all.fasta > ./msa_/{0}_all.msa".format(id1)) ##provide path for Mafft
#print("/home/rahul/IIT_M/conservation/mafft-linux64/mafft.bat --clustalout /home/rahul/IIT_M/conservation/{0}_all.fasta > /home/rahul/IIT_M/conservation/{0}_all.msa".format(i))
os.system("java -jar ./compbio-conservation-1.1.jar -i=./msa_/{0}_all.msa -o=./out_/{0}.AAcons -n -f=RESULT_WITH_NO_ALIGNMENT".format(id1)) # provide path for compbio-conservation
#os.system("mv {}_all.aln ")
fi=GetQueryfromClustalAlignment("./msa_/{0}_all.msa".format(id1))

letters = [ seq[i-1]+str(i) for i in range(1,len(seq)+1) if seq[i-1]!='-' ]
scores  = { x:[] for x in letters }
#print (scores)
aa=open("./out_/{}.AAcons".format(id1)).readlines()
for line in aa:
    try:
        if line.isspace() == False:
            line  = line.replace("\n","").split()
            # get all scores (this is for the gapped sequence
            sc    = list(map(float,line[1:]))
            # print (len(sc))
            # initialize a counter to give position number
            tmpsc = []
            for i,val in enumerate(sc):
                # only consider protein AA, not gaps
                #print (i,val)
                if fi[i] != '-':
                    # combine letter with position counter, append score
                    #print (i,val)
                    tmpsc.append( val )
            for i,val in enumerate(tmpsc):
                scores[ letters[i] ].append(val)
        #print (scores)
    except:
        pass
with open("./AAcon_out/{}.csv".format(id1),"w") as out:
    out.write("AA, KABAT, JORES, SCHNEIDER, SHENKIN, GERSTEIN, TAYLOR_GAPS, TAYLOR_NO_GAPS, VELIBIL, KARLIN, ARMON, THOMPSON, NOT_LANCET, MIRNY, WILLIAMSON, LANDGRAF, SANDER, VALDAR, SMERFS")
    out.write("\n")
    for u,v in scores.items():
        out.write(u+","+",".join([str(x) for x in v]))
        out.write("\n")
# os.system("mv '{}'* ./alignment/".format(id1))
## PSSM SW
df3 = pd.read_csv('./pssm_parsed/'+id1+'.csv')
df2 = pd.DataFrame()
cols = df3.columns[1:-2]
df1 = df3[cols]
for i in range(len(df1)):
    if i-1 < 0:
        x = df1.iloc[i]+df1.iloc[i+1]+df1.iloc[i+2]
    elif i-2 < 0:
        x = df1.iloc[i-1]+df1.iloc[i]+df1.iloc[i+1]+df1.iloc[i+2]
    elif i+2 >= len(df1):
        x = df1.iloc[i]+df1.iloc[i-1]+df1.loc[i-2]
    elif i+1 >= len(df1):
        x = df1.iloc[i+1]+df1.iloc[i]+df1.iloc[i-1]+df1.loc[i-2]
    else:
        x = df1.iloc[i-2]+df1.iloc[i-1]+df1.iloc[i]+df1.iloc[i+1]+df1.iloc[i+2]
    #print(i)
    df2 = df2.append(x, ignore_index = True)
df2['RES'] = df3['RES']
#df2['class'] = df['class']
df2.to_csv(f'./pssm_sw_5/{id1}.csv',index = False)
# STRUCTURE BASED ########################
##########################################################################

#   Download AlphaFold structures
try:
    if not os.path.exists(f"./AlphaFold_out/{id1}.pdb"):
        if len(id1) >= 6:
            cmd = f"https://alphafold.ebi.ac.uk/files/AF-{id1}-F1-model_v4.pdb"
            # cmd = f"https://alphafold.ebi.ac.uk/files/AF-{id1}F1-model_v3.pdb "  ### check for latest link to get alphafold structures
            #cmd = "https://alphafold.ebi.ac.uk/files/AF-A0A6"+id1"-F1-model_v4.pdb"
        else:
            cmd = f"https://files.rcsb.org/download/{id1}.pdb"
        wget_ = f"wget {cmd} -O ./AlphaFold_out/{id1}.pdb"
        out = os.system(wget_)
    else:
        print(f"./AlphaFold_out/{id1}.pdb already exists. Skipping download")

    # DSSP CALCULATIONS
    #cmd = f'/mnt/d/Final/ProB-site/dssp/build/mkdssp /mnt/d/Final/DeepBSRPred/AlphaFold_out/{id1}.pdb /mnt/d/Final/DeepBSRPred/dssp_out/{id1}.dssp'
    cmd = f'/home/eps/prj_envs/Gradio/dssp/build/mkdssp /home/eps/prj_envs/Gradio/DeepBSRPred/AlphaFold_out/{id1}.pdb /home/eps/prj_envs/Gradio/DeepBSRPred/dssp_out/{id1}.dssp'
    print('cmd:', cmd)
    os.system(cmd)
    try:
        f1 = open(f'./dssp_out/{id1}.dssp','r').readlines()
    except FileNotFoundError:
        print(f'check dssp file: {id1}')
    f_out = open(f'./dssp_csv/{id1}.csv','w')
    f_out.writelines('res'+','+'pos'+','+'ASA'+','+'SEC_STR'+'\n')
    for line in f1[28:]:
        try:
            asa = ''
            sec_str = ''
            pos = ''
            res = ''
            #print(line[13:14],line[6:11],line[35:39],line[16:17])
            asa = float(line[35:39].strip())
            pos = int(line[6:11])
            res = line[13:14]
            if res == 'X' or res =='U':
                continue
            if line[16:17].strip() in ['H','G','I']:
                sec_str = 'Helix'
            elif line[16:17].strip() in ['E','B']:
                sec_str ="Sheet"
            elif line[16:17].strip() in ['T']:
                sec_str ='Coil'
            elif line[16:17].strip() in ['S']:
                sec_str ='Coil'
            else:
                sec_str ='Coil'
            f_out.writelines(res+','+str(pos)+','+str(asa)+','+sec_str+'\n')
        except ValueError:
            pass
    f_out.close()
#     CALCULATE ASA percentage
    try:
        df11 = pd.read_csv(f'./dssp_csv/{id1}.csv')
        f1 = open(f'./input_fasta/{id1}.fasta','r').readlines()
        df11['ASA_PER'] = ''
        for i in range(len(df11)):
            try:
                df11['ASA_PER'][i] = round((float(df11['ASA'][i]*100)/float(dict_asa[df11['res'][i]])),2)
            except :
                pass
        # seq = list(f1[1].strip())
        #cls1 = list(f1[2].strip())
        sequences = []
        current_seq = []

        for line in f1:
            if line.startswith(">"):  # 새로운 서열이 시작될 때
                if current_seq:  # 이전 시퀀스 저장
                    sequences.append("".join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.strip())  # 시퀀스 부분 저장

        if current_seq:  # 마지막 서열 저장
            sequences.append("".join(current_seq))
        try:
            df11['seq'] = "".join(sequences)[0:len(df11)]
            #df1['class'] = cls1[0:len(df1)]
        except:
            print("HERE",id1)
        df11.to_csv(f'./ASA_PER_protein/{id1}.csv',index = False)
    except FileNotFoundError:
        print(id1)

    # PDB PARSING
    parser = PDBParser()
    
    # CALCULATE RESIDUE DEPTH
    df11 = pd.read_csv(f'./ASA_PER_protein/{id1}.csv')
    df11['residue_depth'] = ''
    structure = parser.get_structure(id1, f"./AlphaFold_out/{id1}.pdb")
    model = structure[0]
    surface = get_surface(model)
    chain1 = []
    for chains in structure.get_chains():
        chain1.append(chains.id)
    
    rd = ResidueDepth(model)
    RD = []
    
    for chain_id in chain1:
        chain = model[chain_id]
        for residue in chain:
            try:
                # print(f'residue.id: {residue.id}')
                # print(rd[chain1[0],residue.id][0])
                RD.append(round(rd[chain1[0],residue.id][0],2))

            except KeyError:
                pass
                #print(residue)
    
    print(f'len(df(11): {len(df11)}')
    print(f'len(RD): {len(RD)}')
    print(f'id1:::', id1)
    df11['residue_depth'] = RD[0:len(df11)]
    df11.to_csv(f'./ASA_PER_residue_depth/{id1}.csv',index = False)
    # COUNT CONTACT <= 8
    parser = PDBParser()
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
        'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
        'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
        'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    #f_list = glob.glob('./input_fasta/*.fasta')
    #overall = pd.DataFrame()
    try:
        f1 = open(f'./AlphaFold_out/{id1}.pdb').readlines()
        df11 = pd.read_csv(f'./ASA_PER_residue_depth/{id1}.csv')
    except FileNotFoundError:
        print(id1)
    list1 = []
    for i in range(len(f1[0:-1])):
        if(f1[i].startswith('ATOM')):
            count = 0
            residue = f1[i].split()[3].strip()
            atom = f1[i].split()[2].strip()
            if atom == 'CA':
                if residue not in d.keys():
                    continue
                residue_pos = f1[i].split()[5].strip()
                try:
                    x = float(f1[i].split()[6].strip())
                    y = float(f1[i].split()[7].strip())
                    z = float(f1[i].split()[8].strip())
                except:
                    pass
                for j in range(len(f1[0:-1])):
                    if f1[j].startswith('ATOM'):
                        #print('here')
                        residue1 = f1[j].split()[3].strip()
                        atom = f1[j].split()[2].strip()
                        #print(f1[j])
                        if atom == 'CA':
                            #print('here')

                            if residue1 not in d.keys():
                                continue

                            residue_pos1 = f1[j].split()[5].strip()
                            try:
                                x1 = float(f1[j].split()[6].strip())
                                y1 = float(f1[j].split()[7].strip())
                                z1 = float(f1[j].split()[8].strip())
                            except:
                                #print('here')
                                pass
                            distance = math.sqrt((x1-x)**2+(y1-y)**2+(z1-z)**2)
                            #print(x1,y1,z1,distance)
                            if distance<=8:
                                #print('heretoo')
                                count+=1
        #print(count)
        list1.append(count)
#         except:
#             pass


        if len(list1) >len(df11):
            df11['contact_count'] = list1[0:len(df11)]
        elif len(list1)< len(df11):
            for ii in range(len(df11)-len(list1)):
                list1.append(0)
            df11['contact_count'] = list1
        else:
            df11['contact_count'] = list1
        df11.to_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv',index = False)
    
    # HB PLUS
    print("here hbplus")
    os.chdir('./hbplus_out/')
    cmd = f'hbplus -P ../AlphaFold_out/{id1}.pdb'
    os.system(cmd)
    os.chdir('../')
    print("finish hbplus")

    ## PARSE HB PLUS FILES
    d = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
            'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N',
            'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W',
            'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    x=d.keys()
    try:
        h=open(f"./hbplus_out/{id1}.hb2").readlines()[8:]
        df = pd.read_csv(f'./ASA_PER_residue_depth_contact/{id1}.csv')
        df['hb_donor'] = [0]*len(df)
        df['hb_acceptor'] = [0]*len(df)
    except FileNotFoundError:
        print('here line 598',id1)

    for i in range(len(df)):
        res = df['res'][i]
        pos = df['pos'][i]
        for k in h:
            try:
                k=k.rstrip()
                if d[k[:13].split()[0][-3:]] ==res and int(k[:13].split()[0][1:5]) ==pos:
                    df['hb_donor'][i] = df['hb_donor'][i]+1
                if d[k[14:27].split()[0][-3:]] == res and int(k[14:27].split()[0][1:5]) ==pos:
                    df['hb_acceptor'][i] = df['hb_acceptor'][i]+1
            except KeyError:
                pass

    df.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor/{id1}.csv',index = False)
    
    ### NACCESS
    s = parser.get_structure("X",f'./AlphaFold_out/{id1}.pdb')
    model = s[0]
    # rsa_data, asa_data = run_naccess(model , f'./AlphaFold_out/{id1}.pdb')
    rsa_data = run_freesasa(model, f'./AlphaFold_out/{id1}.pdb')
    # print(rsa_data)
    print("="*200)
    num_chains = sum(1 for line in rsa_data if line.startswith("CHAIN"))
    f = rsa_data[9:-(3 + num_chains)]
    df = pd.read_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor/{id1}.csv')
    df['all_abs'] = ''
    df['all_real'] = ''
    df['sc_abs'] = ''
    df['sc_real'] = ''
    df['mc_abs'] = ''
    df['mc_real'] = ''
    df['Non_polar_abs'] = ''
    df['Non_polar_real'] = ''
    df['polar_abs'] = ''
    df['polar_real'] = ''

    for i in range(len(f)):
        try:
            df['all_abs'][i] = f[i].strip().split()[4]
            df['all_real'][i] = f[i].strip().split()[5]
            df['sc_abs'][i] = f[i].strip().split()[6]
            df['sc_real'][i] = f[i].strip().split()[7]
            df['mc_abs'][i] = f[i].strip().split()[8]
            df['mc_real'][i] = f[i].strip().split()[9]
            df['Non_polar_abs'][i] = f[i].strip().split()[10]
            df['Non_polar_real'][i] = f[i].strip().split()[11]
            df['polar_abs'][i] = f[i].strip().split()[12]
            df['polar_real'][i] = f[i].strip().split()[13]
        except IndexError:
            df['all_abs'][i] = f[i].strip().split()[3]
            df['all_real'][i] = f[i].strip().split()[4]
            df['sc_abs'][i] = f[i].strip().split()[5]
            df['sc_real'][i] = f[i].strip().split()[6]
            df['mc_abs'][i] = f[i].strip().split()[7]
            df['mc_real'][i] = f[i].strip().split()[8]
            df['Non_polar_abs'][i] = f[i].strip().split()[9]
            df['Non_polar_real'][i] = f[i].strip().split()[10]
            df['polar_abs'][i] = f[i].strip().split()[11]
            df['polar_real'][i] = f[i].strip().split()[12]
    df.drop(['pos','res'],inplace = True,axis='columns')
    df.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/{id1}.csv',index = False)
except KeyError:
    print()
    a1 = pd.read_csv('ss_avg_aa.csv')
    colss = ['ASA', 'ASA_PER', 'residue_depth', 'contact_count', 'hb_donor', 'hb_acceptor', 'all_abs', 'all_real', 'sc_abs', 'sc_real', 'mc_abs', 'mc_real', 'Non_polar_abs', 'Non_polar_real', 'polar_abs', 'polar_real']
    a= df1[seq].T
    a.columns = colss
    a.to_csv(f'./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/{id1}.csv',index = False)
#    ## merge all data
df_0 = pd.DataFrame()
df_95 = pd.DataFrame()
overall = pd.DataFrame()
#coll = ["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","W","RES","pssm_sum","p","aro","pos_c","neg_c","sul_c","ali","K0","Ht","Hp","P","pHi","pK","Mw","Bl","Rf","Mu","Hnc","Esm","El","Et","Pa","Pb","Pt","Pc","Ca","F","Br","Ra","Ns","aN","aC","aM","V0","Nm","Nl","Hgm","ASAD","ASAN","dASA","dGh","GhD","GhN","dHh","-TdSh","dCph","dGc","dHc","-TdSc","dG","dH","-TdS","v","s","f","Pf-s","GEIM800105","GEIM800108","BIOV880102","GRAR740102","GRAR740103","HOPA770101","ISOY800102","ISOY800103","ISOY800104","JOND750101","JUKT750101","KANM800102","KANM800103","KARP850101","KRIW710101","KRIW790101","LAWE840101","LEVM760101","LIFS790101","LIFS790103","MANP780101","MAXF760102","MAXF760106","MIYS850101","NAGK730102","NAGK730103","BURA740102","ARGP820101","NISK860101","OOBM770103","OOBM850103","CHAM820101","OOBM850105","PONP800102","PONP800103","PONP800107","PRAM900104","RACS770101","RACS770102","RACS820113","CHOC760101","ROBB760105","ROSM880102","SIMZ760101","WOEC730101","CHOP780202","YUTK870102","ZIMJ680101","ZIMJ680102","ZIMJ680105","CHOP780204","ONEK900102","VINM940101","NADH010101","NADH010102","NADH010104","NADH010105","FUKS010102","FUKS010103","KUHL950101","ZHOH040101","ZHOH040102","ZHOH040103","PONJ960101","WOLR790101","OLSK800101","KIDA850101","CORJ870102","CORJ870104","CORJ870106","CORJ870108","MIYS990101","FASG890101","DAYM780201","EISD860101","FASG760101","FASG760102","FAUJ830101","FAUJ880101","FAUJ880106","BIGC670101","ASA","SEC_STR","ASA_PER","seq","class","residue_depth","contact_count","hb_donor","hb_acceptor","all_abs","all_real","sc_abs","sc_real","mc_abs","mc_real","Non_polar_abs","Non_polar_real","polar_abs","polar_real","ASA_PER_AVG","AA"," KABAT"," JORES"," SCHNEIDER"," SHENKIN"," GERSTEIN"," TAYLOR_GAPS"," TAYLOR_NO_GAPS"," VELIBIL"," KARLIN"," ARMON"," THOMPSON"," NOT_LANCET"," MIRNY"," WILLIAMSON"," LANDGRAF"," SANDER"," VALDAR"," SMERFS"]
cols1 = ['RES','GEIM800105', 'F', 'D', 'ISOY800102', 'LAWE840101', 'aN', 'pK', 'Ht', 'p', 'mc_real',  'ali', 'Pf-s', 's', 'MIYS850101', 'V0', 'hb_donor', 'Ra', 'S', 'LIFS790101', ' THOMPSON', 'G', 'Non_polar_abs', 'FASG760101', 'A', 'C', 'E', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'W', 'pssm_sum', 'aro', 'pos_c', 'neg_c', 'sul_c', 'K0', 'Hp',  'pHi', 'Mw', 'Bl']
df = pd.read_csv(f'./window_11/{id1}.csv')
df1 = pd.read_csv(f'./pssm_sw_5/{id1}.csv')### change file
a = pd.read_csv('./ASA_PER_residue_depth_contact_hb_donor_acceptor_naccess/'+id1+'.csv')

aa_index = pd.read_csv(f'./AA_index/{id1}.csv')
df_cons = pd.read_csv(f'./AAcon_out/{id1}.csv')
a['ASA_PER_AVG'] = ''
for i in range(len(a)):
    if i == 0:
        a['ASA_PER_AVG'][i] = (a['ASA_PER'][i]+a['ASA_PER'][i+1]+a['ASA_PER'][i+2])/3
    elif i == 1:
        a['ASA_PER_AVG'][i] = (a['ASA_PER'][i-1]+a['ASA_PER'][i]+a['ASA_PER'][i+1]+a['ASA_PER'][i+2])/4
    elif i == len(a)-1:
        a['ASA_PER_AVG'][i] = (a['ASA_PER'][i-2]+a['ASA_PER'][i-1]+a['ASA_PER'][i])/3
    elif i == len(a)-2:
        a['ASA_PER_AVG'][i] = (a['ASA_PER'][i-2]+a['ASA_PER'][i-1]+a['ASA_PER'][i]+a['ASA_PER'][i+1])/4
    else:
        a['ASA_PER_AVG'][i] = (a['ASA_PER'][i-2]+a['ASA_PER'][i-1]+a['ASA_PER'][i]+a['ASA_PER'][i+1]+a['ASA_PER'][i+2])/5
a['seq'] = df1['RES']
df1['pssm_sum'] = df1[df1.columns[0:-1]].sum(axis = 1,skipna = True)
# a.drop(['pos','res'],inplace = True,axis='columns')
df1 = df1.rename(columns={'P': 'P_df1'})
aa_index = aa_index.rename(columns={'P': 'P_aa_index'})
df3 = pd.concat([df1, df, aa_index, a, df_cons], axis = 1)
df3['P.1'] = ''
df3['V'] = df3['V0']
df3['Y'] = df3['p']
# df3['P.1'] = df3['P']
df3['P'] = df3['P_aa_index']   # aa_index의 'P'를 'P'에 사용
df3['P.1'] = df3['P_df1']       # df1의 'P'를 'P.1'에 사용

# df3['P.1'] = df3['P_aa_index']   # aa_index의 'P'를 'P'에 사용
# df3['P'] = df3['P_df1']       # df1의 'P'를 'P.1'에 사용

def Convert(string):
    list1=[]
    list1[:0]=string
    return list1
data_df1 = pd.DataFrame()
pro_type = 'EC'
cols11 = ['GEIM800105', 'F', 'D', 'ISOY800102', 'LAWE840101', 'aN', 'pK', 'Ht', 'p', 'mc_real', 'V', 'ali', 'Pf-s', 's', 'MIYS850101', 'V0', 'hb_donor', 'Ra', 'S', 'LIFS790101', ' THOMPSON', 'G', 'Non_polar_abs', 'FASG760101', 'A', 'C', 'E', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'T', 'W', 'Y', 'pssm_sum', 'aro', 'pos_c', 'neg_c', 'sul_c', 'K0', 'Hp', 'P.1', 'pHi', 'Mw', 'Bl']
weighted_model = tf.keras.models.load_model(f'./saved_model/{pro_type}_model.h5',compile=False)
scaler = StandardScaler()
f1 = open(f'./input_fasta/{id1}.fasta','r').readlines()
seq1 = f1
#print(seq1)
ll = seq1
dict_temp = {}
for line in range(0,len(ll)-1,2):
    dict_temp[ll[line].split('>')[1].strip()] = ll[line+1].strip()
available = open('./list.txt','r').readlines()
p = .49
EPOCHS = 100
BATCH_SIZE = 20

result_path = f'/home/eps/prj_envs/Gradio/file/{id1}' #upper()
os.makedirs(result_path, exist_ok=True)

for key in dict_temp.keys():
    if key+'\n' in available:
        data_df1 = pd.DataFrame()
        df = pd.read_csv(f'./datasets/merged/{key}.csv')
        train_features = scaler.fit_transform(df[cols1])
        data_df1['Residue'] = df['RES']
        train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
        prediction_label_train = [int(p>=0.5) for p in train_predictions_weighted]
        data_df1['Prediction'] = prediction_label_train

        # Result file
        data_df1.to_csv(f'./results/{id1}_DeepBSRPred_result.csv', index=False) #upper()
        data_df1.to_csv(f'{result_path}/{id1}_DeepBSRPred_result.csv', index=False) #upper()
        print(f"Results are saved in results directory with file name as: {id1}_DeepBSRPred_result.csv file")   #upper()
    else:
        df3 = df3.loc[:,~df3.columns.duplicated()].copy()
        train_features = scaler.fit_transform(df3[cols11])
        data_df1['Residue'] = df3['RES']
        train_predictions_weighted = weighted_model.predict(train_features, batch_size=BATCH_SIZE)
        prediction_label_train = [int(p>=0.5) for p in train_predictions_weighted]

        # Result file
        data_df1['Prediction'] = prediction_label_train
        data_df1.to_csv(f'./results/{id1}_DeepBSRPred_result.csv', index=False) #upper()
        data_df1.to_csv(f'{result_path}/{id1}_DeepBSRPred_result.csv', index=False) #upper()
        print(f"Results are saved in results directory with file name as: {id1}_DeepBSRPred_result.csv file")   #upper()