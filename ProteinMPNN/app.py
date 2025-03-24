# from utils import *
import json
import time
import os
import sys
import glob
import urllib
import shutil
import warnings
import copy
import random
import re
from datetime import datetime

import os.path

import torch
import ray
import jax

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import jax.numpy as jnp
# import tensorflow as tf
import matplotlib.pyplot as plt
# import colabfold as cf
import plotly.graph_objects as go

# import torch.nn as nn
# import torch.nn.functional as F

import tempfile

if "/home/user/app/af_backprop" not in sys.path:
    sys.path.append("/home/user/app/af_backprop")

# local only
if "/home/eps/prj_envs/Gradio/ProteinMPNN/af_backprop" not in sys.path:
    sys.path.append("/home/eps/prj_envs/Gradio/ProteinMPNN/af_backprop")


from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split, Subset
from moleculekit.molecule import Molecule
from ProteinMPNN.af_backprop.alphafold.common import protein
from ProteinMPNN.af_backprop.alphafold.data import pipeline
from ProteinMPNN.af_backprop.alphafold.model import data, config
from ProteinMPNN.af_backprop.alphafold.model import model as afmodel
from ProteinMPNN.af_backprop.alphafold.common import residue_constants

import moleculekit

print(moleculekit.__version__)

from ProteinMPNN.af_backprop.utils import *

sys.path.append("/home/user/app/ProteinMPNN/vanilla_proteinmpnn")
sys.path.append("/home/eps/prj_envs/Gradio/ProteinMPNN/ProteinMPNN/vanilla_proteinmpnn")


# tf.config.set_visible_devices([], "GPU")


def chain_break(idx_res, Ls, length=200):
    # Minkyung's code
    # add big enough number to residue index to indicate chain breaks
    L_prev = 0
    for L_i in Ls[:-1]:
        idx_res[L_prev + L_i:] += length
        L_prev += L_i
    return idx_res


def clear_mem():
    backend = jax.lib.xla_bridge.get_backend()
    for buf in backend.live_buffers():
        buf.delete()


print("Is cuda available ss", torch.cuda.is_available())
# stream = os.popen("nvcc --version")
# output = stream.read()
# print(output)


def setup_af(seq, model_name="model_5_ptm"):
    clear_mem()
    # setup model
    cfg = config.model_config("model_5_ptm")
    cfg.model.num_recycle = 0
    cfg.data.common.num_recycle = 0
    cfg.data.eval.max_msa_clusters = 1
    cfg.data.common.max_extra_msa = 1
    cfg.data.eval.masked_msa_replace_fraction = 0
    cfg.model.global_config.subbatch_size = None
    if os.path.exists("/home/duerr"):
        datadir = "/home/duerr/phd/08_Code/ProteinMPNN"
    else:
        datadir = "/home/eps/prj_envs/Gradio/ProteinMPNN/"
    model_params = data.get_model_haiku_params(model_name=model_name, data_dir=datadir)
    model_runner = afmodel.RunModel(cfg, model_params, is_training=False)
    Ls = [len(s) for s in seq.split("/")]

    seq = re.sub("[^A-Z]", "", seq.upper())
    length = len(seq)
    feature_dict = {
        **pipeline.make_sequence_features(
            sequence=seq, description="none", num_res=length
        ),
        **pipeline.make_msa_features(msas=[[seq]], deletion_matrices=[[[0] * length]]),
    }
    feature_dict["residue_index"] = chain_break(feature_dict["residue_index"], Ls)
    inputs = model_runner.process_features(feature_dict, random_seed=0)

    def runner(seq, opt):
        # update sequence
        inputs = opt["inputs"]
        inputs.update(opt["prev"])
        update_seq(seq, inputs)
        update_aatype(inputs["target_feat"][..., 1:], inputs)

        # mask prediction
        mask = seq.sum(-1)
        inputs["seq_mask"] = inputs["seq_mask"].at[:].set(mask)
        inputs["msa_mask"] = inputs["msa_mask"].at[:].set(mask)
        inputs["residue_index"] = jnp.where(mask == 1, inputs["residue_index"], 0)

        # get prediction
        key = jax.random.PRNGKey(0)
        outputs = model_runner.apply(opt["params"], key, inputs)

        prev = {
            "init_msa_first_row": outputs["representations"]["msa_first_row"][None],
            "init_pair": outputs["representations"]["pair"][None],
            "init_pos": outputs["structure_module"]["final_atom_positions"][None],
        }

        aux = {
            "final_atom_positions": outputs["structure_module"]["final_atom_positions"],
            "final_atom_mask": outputs["structure_module"]["final_atom_mask"],
            "plddt": get_plddt(outputs),
            "pae": get_pae(outputs),
            "inputs": inputs,
            "prev": prev,
        }
        return aux

    return jax.jit(runner), {"inputs": inputs, "params": model_params}


def make_tied_positions_for_homomers(pdb_dict_list):
    my_dict = {}
    for result in pdb_dict_list:
        all_chain_list = sorted(
            [item[-1:] for item in list(result) if item[:9] == "seq_chain"]
        )  # A, B, C, ...
        tied_positions_list = []
        chain_length = len(result[f"seq_chain_{all_chain_list[0]}"])
        for i in range(1, chain_length + 1):
            temp_dict = {}
            for j, chain in enumerate(all_chain_list):
                temp_dict[chain] = [i]  # needs to be a list
            tied_positions_list.append(temp_dict)
        my_dict[result["name"]] = tied_positions_list
    return my_dict


def align_structures(pdb1, pdb2, lenRes, index, random_dir):
    """Take two structure and superimpose pdb1 on pdb2"""
    import Bio.PDB
    import subprocess

    pdb_parser = Bio.PDB.PDBParser(QUIET=True)
    # Get the structures
    ref_structure = pdb_parser.get_structure("ref", pdb1)
    sample_structure = pdb_parser.get_structure("sample", pdb2)

    aligner = Bio.PDB.CEAligner()
    aligner.set_reference(ref_structure)
    aligner.align(sample_structure)

    io = Bio.PDB.PDBIO()
    io.set_structure(ref_structure)
    io.save(f"{random_dir}/outputs/reference.pdb")
    io.set_structure(sample_structure)
    io.save(f"{random_dir}/outputs/out_{index}_aligned.pdb")
    # Doing this to get around biopython CEALIGN bug
    # subprocess.call("pymol -c -Q -r cealign.pml", shell=True)

    return aligner.rms, f"{random_dir}/outputs/reference.pdb", f"{random_dir}/outputs/out_{index}_aligned.pdb"


def save_pdb(outs, filename, LEN):
    """save pdb coordinates"""
    p = {
        "residue_index": outs["inputs"]["residue_index"][0][:LEN],
        "aatype": outs["inputs"]["aatype"].argmax(-1)[0][:LEN],
        "atom_positions": outs["final_atom_positions"][:LEN],
        "atom_mask": outs["final_atom_mask"][:LEN],
    }
    b_factors = 100.0 * outs["plddt"][:LEN, None] * p["atom_mask"]
    p = protein.Protein(**p, b_factors=b_factors)
    pdb_lines = protein.to_pdb(p)
    with open(filename, "w") as f:
        f.write(pdb_lines)


@ray.remote(num_gpus=1, max_calls=1)
def run_alphafold(sequences, num_recycles, random_dir):
    recycles = int(num_recycles)
    RUNNER, OPT = setup_af(sequences[0])
    plddts = []
    paes = []
    for i, sequence in enumerate(sequences):
        SEQ = re.sub("[^A-Z]", "", sequence.upper())
        MAX_LEN = len(SEQ)
        LEN = len(SEQ)

        x = np.array([residue_constants.restype_order.get(aa, -1) for aa in SEQ])
        x = np.pad(x, [0, MAX_LEN - LEN], constant_values=-1)
        x = jax.nn.one_hot(x, 20)

        OPT["prev"] = {
            "init_msa_first_row": np.zeros([1, MAX_LEN, 256]),
            "init_pair": np.zeros([1, MAX_LEN, MAX_LEN, 128]),
            "init_pos": np.zeros([1, MAX_LEN, 37, 3]),
        }

        positions = []

        for r in range(recycles + 1):
            outs = RUNNER(x, OPT)
            outs = jax.tree_map(lambda x: np.asarray(x), outs)
            positions.append(outs["prev"]["init_pos"][0, :LEN])
            OPT["prev"] = outs["prev"]
        plddts.append(outs["plddt"][:LEN])
        paes.append(outs["pae"])
        if os.path.exists("/home/duerr/phd/08_Code/ProteinMPNN"):
            save_pdb(
                outs, f"/home/duerr/phd/08_Code/ProteinMPNN/outputs/out_{i}.pdb", LEN
            )
        else:
            print(f"saving to {random_dir.name}")
            os.system(f"mkdir -p {random_dir.name}/outputs/")
            save_pdb(outs, f"{random_dir.name}/outputs/out_{i}.pdb", LEN)
    return plddts, paes, LEN


def setup_proteinmpnn(model_name="vanilla—v_48_030", backbone_noise=0.00):
    from ProteinMPNN.ProteinMPNN.vanilla_proteinmpnn.protein_mpnn_utils import (
        loss_nll,
        loss_smoothed,
        gather_edges,
        gather_nodes,
        gather_nodes_t,
        cat_neighbors_nodes,
        _scores,
        _S_to_seq,
        tied_featurize,
        parse_PDB,
    )
    from ProteinMPNN.ProteinMPNN.vanilla_proteinmpnn.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

    device = torch.device(
        "cpu"
    )  # torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu") #fix for memory issues
    # ProteinMPNN model name: v_48_002, v_48_010, v_48_020, v_48_030, v_32_002, v_32_010; v_32_020, v_32_030; v_48_010=version with 48 edges 0.10A noise
    # Standard deviation of Gaussian noise to add to backbone atoms
    hidden_dim = 128
    num_layers = 3

    model, model_name = model_name.split("—")
    if os.path.exists("/home/duerr"):
        dir = "/home/duerr/phd/08_Code/ProteinMPNN"
    else:
        dir = "/home/eps/prj_envs/Gradio/ProteinMPNN"

    path_to_model_weights = (
        f"{dir}/ProteinMPNN/{model}_model_weights"
    )

    model_folder_path = path_to_model_weights
    if model_folder_path[-1] != "/":
        model_folder_path = model_folder_path + "/"
    checkpoint_path = model_folder_path + f"{model_name}.pt"
    print("using ProteinMPNN weights from: ", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    noise_level_print = checkpoint["noise_level"]

    model = ProteinMPNN(
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=float(backbone_noise),
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device


def get_pdb(pdb_code="", filepath=""):
    if pdb_code is None or pdb_code == "":
        try:
            return filepath.name
        except AttributeError as e:
            return None
    else:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return f"{pdb_code}.pdb"


def preprocess_mol(pdb_code="", filepath=""):
    print(pdb_code)
    if pdb_code is None or pdb_code == "":
        try:
            print(filepath.name)
            mol = Molecule(filepath.name)
        except AttributeError as e:
            return None
    else:
        os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        print(os.getcwd())
        print(os.listdir())
        print(os.system(f"head -n20 {pdb_code}.pdb"))
        mol = Molecule(f"{pdb_code}.pdb")

    print("print molecule loaded")
    random_dir = f"jobs/{datetime.now().isoformat()}"
    os.system(f"mkdir -p {random_dir}")
    mol.write(f"{random_dir}/original.pdb")
    # clean messy files and only include protein itself
    mol.filter("protein")
    # renumber using moleculekit 0...len(protein)
    df = mol.renumberResidues(returnMapping=True)
    # add proteinMPNN index col which used 1..len(chain), 1...len(chain)
    indexes = []
    for chain, g in df.groupby("chain"):
        j = 1
        for i, row in g.iterrows():
            indexes.append(j)
            j += 1
    df["proteinMPNN_index"] = indexes

    mol.write(f"{random_dir}/cleaned.pdb")

    return f"{random_dir}/cleaned.pdb", df, f"{random_dir}/original.pdb"


def assign_sasa(mol):
    from moleculekit.projections.metricsasa import MetricSasa

    metr = MetricSasa(mode="residue", filtersel="protein")
    sasaR = metr.project(mol)[0]
    is_prot = mol.atomselect("protein")
    resids = pd.DataFrame.from_dict({"resid": mol.resid, "is_prot": is_prot})
    new_masses = []
    i_without_non_prot = 0
    for i, g in resids.groupby((resids["resid"].shift() != resids["resid"]).cumsum()):
        if g["is_prot"].unique()[0] == True:
            g["sasa"] = sasaR[i_without_non_prot]
            i_without_non_prot += 1
        else:
            g["sasa"] = 0
        new_masses.extend(list(g.sasa))
    return np.array(new_masses)


def process_atomsel(atomsel):
    """everything lowercase and replace some keywords not relevant for protein design"""
    atomsel = re.sub("sasa", "mass", atomsel, flags=re.I)
    atomsel = re.sub("plddt", "beta", atomsel, flags=re.I)
    return atomsel


def make_fixed_positions_dict(original_file, atomsel, residue_index_df):
    # we use the uploaded file for the selection
    print("fixed_pos using", original_file)
    print(os.system(f"head -n10 {original_file}"))
    mol = Molecule(original_file)
    # use index for selection as resids will change

    # set sasa to 0 for all non protein atoms (all non protein atoms are deleted later)
    mol.masses = assign_sasa(mol)
    print(mol.masses.shape)
    print(assign_sasa(mol).shape)
    atomsel = process_atomsel(atomsel)
    selected_residues = mol.get("index", atomsel)

    # clean up
    mol.filter("protein")
    mol.renumberResidues()
    # based on selected index now get resids
    selected_residues = [str(i) for i in selected_residues]
    if len(selected_residues) == 0:
        return None, []
    selected_residues_str = " ".join(selected_residues)
    selected_residues = set(mol.get("resid", sel=f"index {selected_residues_str}"))

    # use the proteinMPNN index nomenclature to assemble fixed_positions_dict
    fixed_positions_df = residue_index_df[
        residue_index_df["new_resid"].isin(selected_residues)
    ]

    chains = set(mol.get("chain", sel="all"))
    fixed_position_dict = {"cleaned": {}}
    # store the selected residues in a list for the visualization later with cleaned.pdb
    selected_residues = list(fixed_positions_df["new_resid"])

    for c in chains:
        fixed_position_dict["cleaned"][c] = []

    for i, row in fixed_positions_df.iterrows():
        fixed_position_dict["cleaned"][row["chain"]].append(row["proteinMPNN_index"])
    return fixed_position_dict, selected_residues


def update(
    inp,
    file,
    designed_chain,
    fixed_chain,
    homomer,
    num_seqs,
    sampling_temp,
    model_name,
    backbone_noise,
    omit_AAs,
    atomsel,
):
    from ProteinMPNN.ProteinMPNN.vanilla_proteinmpnn.protein_mpnn_utils import (
        loss_nll,
        loss_smoothed,
        gather_edges,
        gather_nodes,
        gather_nodes_t,
        cat_neighbors_nodes,
        _scores,
        _S_to_seq,
        tied_featurize,
        parse_PDB,
    )
    from ProteinMPNN.ProteinMPNN.vanilla_proteinmpnn.protein_mpnn_utils import StructureDataset, StructureDatasetPDB, ProteinMPNN

    # pdb_path = get_pdb(pdb_code=inp, filepath=file)

    pdb_path, mol_index, path_unprocessed = preprocess_mol(pdb_code=inp, filepath=file)

    print("done processing mol")
    if pdb_path == None:
        return "Error processing PDB"

    model, device = setup_proteinmpnn(
        model_name=model_name, backbone_noise=float(backbone_noise)
    )

    if designed_chain == "":
        designed_chain_list = []
    else:
        designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

    if fixed_chain == "":
        fixed_chain_list = []
    else:
        fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

    chain_list = list(set(designed_chain_list + fixed_chain_list))
    num_seq_per_target = int(num_seqs)
    save_score = 0  # 0 for False, 1 for True; save score=-log_prob to npy files
    save_probs = (
        0  # 0 for False, 1 for True; save MPNN predicted probabilites per position
    )
    score_only = 0  # 0 for False, 1 for True; score input backbone-sequence pairs
    # 0 for False, 1 for True; output conditional probabilities p(s_i given the rest of the sequence and backbone)
    conditional_probs_only = 0
    # 0 for False, 1 for True; if true output conditional probabilities p(s_i given backbone)
    conditional_probs_only_backbone = 0

    batch_size = 1  # Batch size; can set higher for titan, quadro GPUs, reduce this if running out of GPU memory
    max_length = 20000  # Max sequence length

    out_folder = "."  # Path to a folder to output sequences, e.g. /home/out/
    jsonl_path = ""  # Path to a folder with parsed pdb into jsonl

    if omit_AAs == "":
        # Specify which amino acids should be omitted in the generated sequence, e.g. 'AC' would omit alanine and cystine.
        omit_AAs = "X"

    pssm_multi = 0.0  # A value between [0.0, 1.0], 0.0 means do not use pssm, 1.0 ignore MPNN predictions
    pssm_threshold = 0.0  # A value between -inf + inf to restric per position AAs
    pssm_log_odds_flag = 0  # 0 for False, 1 for True
    pssm_bias_flag = 0  # 0 for False, 1 for True

    folder_for_outputs = out_folder

    sampling_temp = float(sampling_temp)

    NUM_BATCHES = num_seq_per_target // batch_size
    BATCH_COPIES = batch_size
    temperatures = [sampling_temp]
    omit_AAs_list = omit_AAs
    alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    omit_AAs_np = np.array([AA in omit_AAs_list for AA in alphabet]).astype(np.float32)

    chain_id_dict = None
    if atomsel == "":
        fixed_positions_dict, selected_residues = None, []
    else:
        fixed_positions_dict, selected_residues = make_fixed_positions_dict(path_unprocessed,
                                                                            atomsel, mol_index
                                                                            )

    pssm_dict = None
    omit_AA_dict = None
    bias_AA_dict = None

    bias_by_res_dict = None
    bias_AAs_np = np.zeros(len(alphabet))

    ###############################################################
    pdb_dict_list = parse_PDB(pdb_path, input_chain_list=chain_list)
    dataset_valid = StructureDatasetPDB(
        pdb_dict_list, truncate=None, max_length=max_length
    )
    if homomer:
        tied_positions_dict = make_tied_positions_for_homomers(pdb_dict_list)
    else:
        tied_positions_dict = None

    chain_id_dict = {}
    chain_id_dict[pdb_dict_list[0]["name"]] = (designed_chain_list, fixed_chain_list)
    with torch.no_grad():
        for ix, prot in enumerate(dataset_valid):
            score_list = []
            all_probs_list = []
            all_log_probs_list = []
            S_sample_list = []
            batch_clones = [copy.deepcopy(prot) for i in range(BATCH_COPIES)]
            (
                X,
                S,
                mask,
                lengths,
                chain_M,
                chain_encoding_all,
                chain_list_list,
                visible_list_list,
                masked_list_list,
                masked_chain_length_list_list,
                chain_M_pos,
                omit_AA_mask,
                residue_idx,
                dihedral_mask,
                tied_pos_list_of_lists_list,
                pssm_coef,
                pssm_bias,
                pssm_log_odds_all,
                bias_by_res_all,
                tied_beta,
            ) = tied_featurize(
                batch_clones,
                device,
                chain_id_dict,
                fixed_positions_dict,
                omit_AA_dict,
                tied_positions_dict,
                pssm_dict,
                bias_by_res_dict,
            )
            pssm_log_odds_mask = (
                pssm_log_odds_all > pssm_threshold
            ).float()  # 1.0 for true, 0.0 for false
            name_ = batch_clones[0]["name"]

            randn_1 = torch.randn(chain_M.shape, device=X.device)
            log_probs = model(
                X,
                S,
                mask,
                chain_M * chain_M_pos,
                residue_idx,
                chain_encoding_all,
                randn_1,
            )
            mask_for_loss = mask * chain_M * chain_M_pos
            scores = _scores(S, log_probs, mask_for_loss)
            native_score = scores.cpu().data.numpy()
            message = ""
            seq_list = []
            seq_recovery = []
            seq_score = []
            for temp in temperatures:
                for j in range(NUM_BATCHES):
                    randn_2 = torch.randn(chain_M.shape, device=X.device)
                    if tied_positions_dict == None:
                        sample_dict = model.sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=float(temp),
                            omit_AAs_np=omit_AAs_np,
                            bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=bool(pssm_log_odds_flag),
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=bool(pssm_bias_flag),
                            bias_by_res=bias_by_res_all,
                        )
                        S_sample = sample_dict["S"]
                    else:
                        sample_dict = model.tied_sample(
                            X,
                            randn_2,
                            S,
                            chain_M,
                            chain_encoding_all,
                            residue_idx,
                            mask=mask,
                            temperature=float(temp),
                            omit_AAs_np=omit_AAs_np,
                            bias_AAs_np=bias_AAs_np,
                            chain_M_pos=chain_M_pos,
                            omit_AA_mask=omit_AA_mask,
                            pssm_coef=pssm_coef,
                            pssm_bias=pssm_bias,
                            pssm_multi=pssm_multi,
                            pssm_log_odds_flag=bool(pssm_log_odds_flag),
                            pssm_log_odds_mask=pssm_log_odds_mask,
                            pssm_bias_flag=bool(pssm_bias_flag),
                            tied_pos=tied_pos_list_of_lists_list[0],
                            tied_beta=tied_beta,
                            bias_by_res=bias_by_res_all,
                        )
                        # Compute scores
                        S_sample = sample_dict["S"]
                    log_probs = model(
                        X,
                        S_sample,
                        mask,
                        chain_M * chain_M_pos,
                        residue_idx,
                        chain_encoding_all,
                        randn_2,
                        use_input_decoding_order=True,
                        decoding_order=sample_dict["decoding_order"],
                    )
                    mask_for_loss = mask * chain_M * chain_M_pos
                    scores = _scores(S_sample, log_probs, mask_for_loss)
                    scores = scores.cpu().data.numpy()
                    all_probs_list.append(sample_dict["probs"].cpu().data.numpy())
                    all_log_probs_list.append(log_probs.cpu().data.numpy())
                    S_sample_list.append(S_sample.cpu().data.numpy())
                    for b_ix in range(BATCH_COPIES):
                        masked_chain_length_list = masked_chain_length_list_list[b_ix]
                        masked_list = masked_list_list[b_ix]
                        seq_recovery_rate = torch.sum(
                            torch.sum(
                                torch.nn.functional.one_hot(S[b_ix], 21)
                                * torch.nn.functional.one_hot(S_sample[b_ix], 21),
                                axis=-1,
                            )
                            * mask_for_loss[b_ix]
                        ) / torch.sum(mask_for_loss[b_ix])
                        seq = _S_to_seq(S_sample[b_ix], chain_M[b_ix])
                        score = scores[b_ix]
                        score_list.append(score)
                        native_seq = _S_to_seq(S[b_ix], chain_M[b_ix])
                        if b_ix == 0 and j == 0 and temp == temperatures[0]:
                            start = 0
                            end = 0
                            list_of_AAs = []
                            for mask_l in masked_chain_length_list:
                                end += mask_l
                                list_of_AAs.append(native_seq[start:end])
                                start = end
                            native_seq = "".join(
                                list(np.array(list_of_AAs)[np.argsort(masked_list)])
                            )
                            l0 = 0
                            for mc_length in list(
                                np.array(masked_chain_length_list)[
                                    np.argsort(masked_list)
                                ]
                            )[:-1]:
                                l0 += mc_length
                                native_seq = native_seq[:l0] + "/" + native_seq[l0:]
                                l0 += 1
                            sorted_masked_chain_letters = np.argsort(
                                masked_list_list[0]
                            )
                            print_masked_chains = [
                                masked_list_list[0][i]
                                for i in sorted_masked_chain_letters
                            ]
                            sorted_visible_chain_letters = np.argsort(
                                visible_list_list[0]
                            )
                            print_visible_chains = [
                                visible_list_list[0][i]
                                for i in sorted_visible_chain_letters
                            ]
                            native_score_print = np.format_float_positional(
                                np.float32(native_score.mean()),
                                unique=False,
                                precision=4,
                            )
                            line = ">{}, score={}, fixed_chains={}, designed_chains={}, model_name={}\n{}\n".format(
                                name_,
                                native_score_print,
                                print_visible_chains,
                                print_masked_chains,
                                model_name,
                                native_seq,
                            )
                            message += f"{line}\n"
                        start = 0
                        end = 0
                        list_of_AAs = []
                        for mask_l in masked_chain_length_list:
                            end += mask_l
                            list_of_AAs.append(seq[start:end])
                            start = end

                        seq = "".join(
                            list(np.array(list_of_AAs)[np.argsort(masked_list)])
                        )
                        # add non designed chains to predicted sequence
                        l0 = 0
                        for mc_length in list(
                            np.array(masked_chain_length_list)[np.argsort(masked_list)]
                        )[:-1]:
                            l0 += mc_length
                            seq = seq[:l0] + "/" + seq[l0:]
                            l0 += 1
                        score_print = np.format_float_positional(
                            np.float32(score), unique=False, precision=4
                        )
                        seq_rec_print = np.format_float_positional(
                            np.float32(seq_recovery_rate.detach().cpu().numpy()),
                            unique=False,
                            precision=4,
                        )
                        chain_s = ""
                        if len(visible_list_list[0]) > 0:
                            chain_M_bool = chain_M.bool()
                            not_designed = _S_to_seq(S[b_ix], ~chain_M_bool[b_ix])

                            labels = (
                                chain_encoding_all[b_ix][~chain_M_bool[b_ix]]
                                .detach()
                                .cpu()
                                .numpy()
                            )

                            for c in set(labels):
                                chain_s += "/"
                                nd_mask = labels == c
                                for i, x in enumerate(not_designed):
                                    if nd_mask[i]:
                                        chain_s += x
                        seq_recovery.append(seq_rec_print)
                        seq_score.append(score_print)
                        line = (
                            ">T={}, sample={}, score={}, seq_recovery={}\n{}\n".format(
                                temp, b_ix, score_print, seq_rec_print, seq
                            )
                        )
                        seq_list.append(seq + chain_s)
                        message += f"{line}\n"
    if fixed_positions_dict != None:
        message += f"\nfixed positions:* {fixed_positions_dict['cleaned']} \n\n*uses CHAIN:[1..len(chain)] residue numbering"
    # somehow sequences still contain X, remove again
    for i, x in enumerate(seq_list):
        for aa in omit_AAs:
            seq_list[i] = x.replace(aa, "")
    all_probs_concat = np.concatenate(all_probs_list)
    all_log_probs_concat = np.concatenate(all_log_probs_list)
    np.savetxt("all_probs_concat.csv", all_probs_concat.mean(0).T, delimiter=",")
    np.savetxt(
        "all_log_probs_concat.csv",
        np.exp(all_log_probs_concat).mean(0).T,
        delimiter=",",
    )
    S_sample_concat = np.concatenate(S_sample_list)
    fig = px.imshow(
        np.exp(all_log_probs_concat).mean(0).T,
        #title="Amino acid probabilities",
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )
    fig.update_xaxes(side="top")

    fig_tadjusted = px.imshow(
        all_probs_concat.mean(0).T,
        #title="T adjusted probabilities",
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )

    fig_tadjusted.update_xaxes(side="top")
    seq_dict = {"seq_list": seq_list, "recovery": seq_recovery, "seq_score": seq_score}
    return (
        message,
        fig,
        fig_tadjusted,
        gr.update(value="all_log_probs_concat.csv", visible=True),
        gr.update(value="all_probs_concat.csv", visible=True),
        pdb_path,
        gr.update(choices=seq_list),
        selected_residues,
        seq_dict,
    )


def update_AF(seq_dict, pdb, num_recycles, selectedResidues):

    # # run alphafold using ray
    # plddts, pae, num_res = run_alphafold(
    #    startsequence, num_recycles
    # )
    allSeqs = seq_dict["seq_list"]
    lenSeqs = len(allSeqs)
    if len(allSeqs[0]) > 700:
        return (
            """
            <div class="p-4 mb-4 text-sm text-yellow-700 bg-orange-50 rounded-lg" role="alert">
  <span class="font-medium">Sorry!</span> Currently only small proteins can be run in the server in order to reduce wait time. Try a protein <700 aa. Bigger proteins you can run on <a href="https://github.com/sokrypton/colabfold">ColabFold</a>
</div>
""",
            plt.figure(),
            plt.figure(),
        )
    random_dir = tempfile.TemporaryDirectory()

    plddts, paes, num_res = ray.get(run_alphafold.remote(allSeqs, num_recycles, random_dir))

    sequences = {}
    for i in range(lenSeqs):
        rms, input_pdb, aligned_pdb = align_structures(
            pdb, f"{random_dir.name}/outputs/out_{i}.pdb", num_res, i, random_dir.name
        )
        sequences[i] = {
            "Seq": i,
            "RMSD": f"{rms:.2f}",
            "Score": seq_dict["seq_score"][i],
            "Recovery": seq_dict["recovery"][i],
            "Mean pLDDT": f"{np.mean(plddts[i]):.4f}",
        }
    results = pd.DataFrame.from_dict(sequences, orient="index")
    print(results)
    plots = []
    for index, plddts_val in enumerate(plddts):
        # if recycle == 0 or recycle == len(plddts) - 1:
        #     visible = True
        # else:
        #     visible = "legendonly"
        visible = True
        plots.append(
            go.Scatter(
                x=np.arange(len(plddts_val)),
                y=plddts_val,
                hovertemplate="<i>pLDDT</i>: %{y:.2f} <br><i>Residue index:</i> %{x}<br>Sequence "
                + str(index),
                name=f"seq {index}",
                visible=visible,
            )
        )
    plotAF_plddt = go.Figure(data=plots)
    plotAF_plddt.update_layout(
        title="pLDDT",
        xaxis_title="Residue index",
        yaxis_title="pLDDT",
        height=500,
        template="simple_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.99),
    )
    pae_plots = []
    for i, pae in enumerate(paes):
        plt.figure()
        plt.title(f"Predicted Aligned Error sequence {i}")
        Ln = pae.shape[0]
        plt.imshow(pae, cmap="bwr", vmin=0, vmax=30, extent=(0, Ln, Ln, 0))
        plt.colorbar()
        plt.xlabel("Scored residue")
        plt.ylabel("Aligned residue")
        plt.savefig(f"ProteinMPNN/outputs/pae_plot_{i}.png", dpi=300)
        plt.close()
        pae_plots.append(f"ProteinMPNN/outputs/pae_plot_{i}.png")
    # doesnt work (likely because too large)
    # plotAF_pae = px.imshow(
    #     pae,
    #     labels=dict(x="Scored residue", y="Aligned residue", color=""),
    #     template="simple_white",
    #     y=np.arange(len(plddts_val)),
    # )
    # plotAF_pae.write_html("test.html")
    # plotAF_pae.update_layout(title="Predicted Aligned Error", template="simple_white")

    return (
        molecule(
            input_pdb,
            aligned_pdb,
            lenSeqs,
            num_res,
            selectedResidues,
            allSeqs,
            sequences,
            random_dir.name
        ),
        plotAF_plddt,
        pae_plots,
        results,
    )


def read_mol(molpath):
    with open(molpath, "r") as fp:
        lines = fp.readlines()
    mol = ""
    for l in lines:
        mol += l
    return mol


def molecule(
    input_pdb, aligned_pdb, lenSeqs, num_res, selectedResidues, allSeqs, sequences, random_dir
):

    mol = read_mol(f"{random_dir}/outputs/reference.pdb")
    options = ""
    pred_mol = "["
    seqdata = "{"
    selected = "selected"
    for i in range(lenSeqs):
        seqdata += (
            str(i)
            + ': { "score": '
            + sequences[i]["Score"]
            + ', "rmsd": '
            + sequences[i]["RMSD"]
            + ', "recovery": '
            + sequences[i]["Recovery"]
            + ', "plddt": '
            + sequences[i]["Mean pLDDT"]
            + ', "seq":"'
            + allSeqs[i]
            + '"}'
        )
        # RMSD {sequences[i]["RMSD"]}, score {sequences[i]["Score"]}, recovery {sequences[i]["Recovery"]} pLDDT {sequences[i]["Mean pLDDT"]}
        options += f'<option {selected} value="{i}">sequence {i} </option>'
        p = f"{random_dir}/outputs/out_{i}_aligned.pdb"
        pred_mol += f"`{read_mol(p)}`"
        selected = ""
        if i != lenSeqs - 1:
            pred_mol += ","
            seqdata += ","
    pred_mol += "]"
    seqdata += "}"

    x = (
        """<!DOCTYPE html>
        <html>
        <head>    
    <meta http-equiv="content-type" content="text/html; charset=UTF-8" />
     <link rel="stylesheet" href="https://unpkg.com/flowbite@1.4.5/dist/flowbite.min.css" />
    <style>
    body{
        font-family:sans-serif
    }
    .mol-container {
    width: 100%;
    height: 700px;
    position: relative;
    }
    .space-x-2 > * + *{
        margin-left: 0.5rem;
    }
    .p-1{
        padding:0.5rem;
    }
    .w-4{
        width:1rem;
    }
    .h-4{
        height:1rem;
    }
    .mt-4{
        margin-top:1rem;
    }
    .mol-container select{
        background-image:None;
    }
    .click_button { /* elem_id="click_button"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
        background-color: #003399 !important; /* 배경색 변경 */
        color: white !important; /* 글자색 변경 */
        border-radius: 10px !important; /* (선택 사항) 버튼 모서리 둥글게 */
        border-color: #003399; /* 테두리 색상 설정 */
        font-size: 15px !important; /* 글자 크기 20px로 설정 */
    }


    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery/3.6.3/jquery.min.js" integrity="sha512-STof4xm1wgkfm7heWqFJVn58Hm3EtS31XFaagaa8VMReCXAkQnJZ+jEy8PCC/iT18dFy95WcExNHFTqLyp72eQ==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>
    <script src="https://3Dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>  
    <div class="max-w-2xl flex items-center space-x-2 py-3">
        <label for="seq"
            class=" text-right whitespace-nowrap block text-base font-medium text-white-900 dark:text-white-400" style="color:white" >시퀀스 선택</label>
        <select id="seq"
            class="bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-blue-500 focus:border-blue-500 block w-full p-2.5 dark:bg-gray-700 dark:border-gray-600 dark:placeholder-gray-400 dark:text-white dark:focus:ring-blue-500 dark:focus:border-blue-500">
            """
        + options
        + """
        </select>
    </div>
    <div class="font-mono bg-gray-100 py-3 px-2  font-sm rounded">
        <code>> seq <span id="id"></span>, score <span id="score"></span>, RMSD <span id="seqrmsd"></span>, Recovery
            <span id="recovery"></span>, pLDDT <span id="plddt"></span></code><br>
        <p id="seqText" class="max-w-4xl font-xs block" style="word-break: break-all;">

        </p>
    </div>
    <div id="container" class="mol-container"></div>
    <div class="flex items-center">
        <div class="px-4 pt-2">
        <label for="sidechain" class="relative inline-flex items-center mb-4 cursor-pointer ">
            <input  id="sidechain" type="checkbox" class="sr-only peer">
            <div class="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            <span class="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300" style="color:white">사이드 체인 보기</span>
          </label>
        </div>
        <div class="px-4 pt-2">
        <label for="startstructure" class="relative inline-flex items-center mb-4 cursor-pointer ">
            <input  id="startstructure" type="checkbox" class="sr-only peer" checked>
            <div class="w-11 h-6 bg-gray-200 rounded-full peer peer-focus:ring-4 peer-focus:ring-blue-300 dark:peer-focus:ring-blue-800 dark:bg-gray-700 peer-checked:after:translate-x-full peer-checked:after:border-white after:absolute after:top-0.5 after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all dark:border-gray-600 peer-checked:bg-blue-600"></div>
            <span class="ml-3 text-sm font-medium text-gray-900 dark:text-gray-300" style="color:white">백본 구조 보기</span>
          </label>
        </div>
        <button type="button" class="flex text-gray-900 bg-white hover:bg-gray-100 border border-gray-200 focus:ring-4 focus:outline-none focus:ring-gray-100 font-medium rounded-lg text-sm px-5 py-2.5 text-center inline-flex items-center dark:focus:ring-gray-600 dark:bg-gray-800 dark:border-gray-700 dark:text-white dark:hover:bg-gray-700 mr-2 mb-2 click_button" id="download">
                    <svg class="w-6 h-6 mr-2 -ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"></path></svg>
                    예측 구조 다운로드
                  </button>
            </div>       
            <div class="text-sm" style="color:white">
            <div> RMSD AlphaFold vs. native: <span id="rmsd"></span> Å는 CEAlign을 사용하여 정렬된 단편에서 계산됨</div>
                                    </div>
            <div class="text-sm flex items-start" style="color:white">
                <div class="w-1/2">
                        
                            <div class="font-medium mt-4 flex items-center space-x-2"><b>재설계된 서열의 AF2 모델</b></div>
                            <div>AlphaFold 모델 신뢰도:</div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4" style="background-color: rgb(0, 83, 214);">&nbsp;</span><span class="legendlabel">매우 높음
                                    (pLDDT &gt; 90)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4" style="background-color: rgb(101, 203, 243);">&nbsp;</span><span class="legendlabel">높음
                                    (90 &gt; pLDDT &gt; 70)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4" style="background-color: rgb(255, 219, 19);">&nbsp;</span><span class="legendlabel">낮음 (70 &gt;
                                    pLDDT &gt; 50)</span></div>
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4" style="background-color: rgb(255, 125, 69);">&nbsp;</span><span class="legendlabel">매우 낮음
                                    (pLDDT &lt; 50)</span></div>
                            <div class="row column legendDesc"> AlphaFold는 0에서 100 사이의 잔기별 신뢰도 점수(pLDDT)를 생성합니다. pLDDT가 50 미만인 일부 영역은 단독으로는 구조화되지 않을 수 있습니다.
                            </div>
                        </div>
                        <div class="w-1/2">
                            <div class="font-medium mt-4 flex items-center space-x-2"><b>백본 구조 </b><span class="w-4 h-4 bg-gray-300 inline-flex" ></span></div>
                            
                            <div class="flex space-x-2 py-1"><span class="w-4 h-4" style="background-color:hotpink" >&nbsp;</span><span class="legendlabel">고정된 위치</span></div>

                        </div>
                    </div>
            <script>

              function drawStructures(i, selectedResidues) {
            $("#rmsd").text(seqs[i]["rmsd"])
            $("#seqText").text(seqs[i]["seq"])
            $("#seqrmsd").text(seqs[i]["rmsd"])
            $("#id").text(i)
            $("#score").text(seqs[i]["score"])
            $("#recovery").text(seqs[i]["recovery"])
            $("#plddt").text(seqs[i]["plddt"])

            viewer = $3Dmol.createViewer(element, config);
            viewer.addModel(data[i], "pdb");
            viewer.addModel(pdb, "pdb");



            viewer.getModel(1).setStyle({}, { cartoon: { colorscheme: { prop: "resi", map: colors } } })
            viewer.getModel(0).setStyle({}, { cartoon: { colorfunc: colorAlpha } });
            viewer.zoomTo();
            viewer.render();
            viewer.zoom(0.8, 2000);
            viewer.getModel(0).setHoverable({}, true,
                function (atom, viewer, event, container) {
                    if (!atom.label) {
                        atom.label = viewer.addLabel(atom.resn + atom.resi + " pLDDT=" + atom.b, { position: atom, backgroundColor: "mintcream", fontColor: "black" });
                    }
                },
                function (atom, viewer) {
                    if (atom.label) {
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }
                }
            );
        }
        let viewer = null;
        let voldata = null;
        let element = null;
        let config = null;
        let currentIndex = 0;
            let seqs = """
        + seqdata
        + """
               let data = """
        + pred_mol
        + """  
                let pdb = `"""
        + mol
        + """`  
         var selectedResidues = """
        + f"{selectedResidues}"
        + """
                //AlphaFold code from https://gist.github.com/piroyon/30d1c1099ad488a7952c3b21a5bebc96
                let colorAlpha = function (atom) {
                    if (atom.b < 50) {
                        return "OrangeRed";
                    } else if (atom.b < 70) {
                        return "Gold";
                    } else if (atom.b < 90) {
                        return "MediumTurquoise";
                    } else {
                        return "Blue";
                    }
                };
               
                let colors = {}
                for (let i=0; i<"""
        + str(num_res)
        + """;i++){
                if (selectedResidues.includes(i)){
                    colors[i]="hotpink"
                }else{
                    colors[i]="lightgray"
                }}

                let colorFixedSidechain = function(atom){
                                if (selectedResidues.includes(atom.resi)){
                                    return "hotpink"
                                }else if (atom.elem == "O"){
                                    return "red"
                                }else if (atom.elem == "N"){
                                    return "blue"
                                }else if (atom.elem == "S"){
                                    return "yellow"
                                }else{
                                    return "lightgray"
                                }
                            }

             $(document).ready(function () {
                element = $("#container");
                config = { backgroundColor: "white" };
                //viewer.ui.initiateUI();
 
                drawStructures(currentIndex, selectedResidues)
                $("#sidechain").change(function () {
                    if (this.checked) {
                        BB = ["C", "O", "N"]
                        
                        if ($("#startstructure").prop("checked")) {
                            viewer.getModel(0).setStyle( {"and": [{resn: ["GLY", "PRO"], invert: true},{atom: BB, invert: true},]},{stick: {colorscheme: "WhiteCarbon", radius: 0.3}, cartoon: { colorfunc: colorAlpha }});
                            viewer.getModel(1).setStyle( {"and": [{resn: ["GLY", "PRO"], invert: true},{atom: BB, invert: true},]},{stick: {colorfunc:colorFixedSidechain, radius: 0.3}, cartoon: {colorscheme:{prop:"resi",map:colors} }});
                        }else{
                            viewer.getModel(0).setStyle( {"and": [{resn: ["GLY", "PRO"], invert: true},{atom: BB, invert: true},]},{stick: {colorscheme: "WhiteCarbon", radius: 0.3}, cartoon: { colorfunc: colorAlpha }});
                            viewer.getModel(1).setStyle();                        
                        }
                        
                        viewer.render()
                    } else {
                        if ($("#startstructure").prop("checked")) {
                        viewer.getModel(0).setStyle({cartoon: { colorfunc: colorAlpha }});
                        viewer.getModel(1).setStyle({cartoon: {colorscheme:{prop:"resi",map:colors} }});
                        }else{
                            viewer.getModel(0).setStyle({cartoon: { colorfunc: colorAlpha }});
                            viewer.getModel(1).setStyle();
                            }
                        viewer.render()
                    }
                });
                $("#seq").change(function () {
                    drawStructures(this.value, selectedResidues)
                    currentIndex = this.value
                    $("#sidechain").prop( "checked", false );
                    $("#startstructure").prop( "checked", true );
                });
                $("#startstructure").change(function () {
                    if (this.checked) {
                         $("#sidechain").prop( "checked", false );
                       viewer.getModel(1).setStyle({},{cartoon: {colorscheme:{prop:"resi",map:colors} } })
                       viewer.getModel(0).setStyle({}, { cartoon: { colorfunc: colorAlpha } });
                       viewer.render()
                    } else {
                        $("#sidechain").prop( "checked", false );
                       viewer.getModel(1).setStyle({},{})
                       viewer.getModel(0).setStyle({}, { cartoon: { colorfunc: colorAlpha } });
                        viewer.render()
                    }
                });
                $("#download").click(function () {
                    download("outputs/out_" + currentIndex + "_aligned.pdb", data[currentIndex]);
                })
        });
        function download(filename, text) {
            var element = document.createElement("a");
            element.setAttribute("href", "data:text/plain;charset=utf-8," + encodeURIComponent(text));
            element.setAttribute("download", filename);
            element.style.display = "none";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);
        }
        </script>
        </body></html>"""
    )

    return f"""<iframe style="width: 800px; height: 1300px" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{x}'></iframe>"""


def set_examples(example):
    (
        label,
        inp,
        designed_chain,
        fixed_chain,
        homomer,
        num_seqs,
        sampling_temp,
        atomsel,
    ) = example
    return [
        label,
        inp,
        designed_chain,
        fixed_chain,
        homomer,
        gr.update(value=num_seqs),
        gr.update(value=sampling_temp),
        atomsel,
    ]

css = """
#click_button { /* elem_id="click_button"인 버튼 선택, 메뉴 내내 화면 버튼 적용 */
    background-color: #003399 !important; /* 배경색 변경 */
    color: white !important; /* 글자색 변경 */
    /* width: 250px !important;  가로 사이즈 변경 */
    border-radius: 10px !important; /* (선택 사항) 버튼 모서리 둥글게 */
    padding: 10px 20px !important; /* (선택 사항) 버튼 내부 여백 조정 padding (상하, 좌우)*/
    margin-left: auto !important; /* 가로 가운데 정렬 */
    margin-right: auto !important; /* 가로 가운데 정렬 */
    display: block !important; /* 블록 요소 설정 */
    font-size: 15px !important; /* 글자 크기 20px로 설정 */
    text-align: center !important; /* 텍스트 가운데 정렬 */
}
"""

# proteinMPNN = gr.Blocks()
def get_mpnn_ui():
    with gr.Blocks() as proteinMPNN:
        gr.Markdown("# Input")
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    inp = gr.Textbox(
                        placeholder="ex) P01116, P68871 ...", label="단백질 ID 입력"
                    )
                with gr.Column():
                    file = gr.File(file_count="single")
            with gr.Column():
                with gr.Column():
                    designed_chain = gr.Textbox(value="A", label="대상 체인")
                    fixed_chain = gr.Textbox(
                        placeholder="다중 체인을 연결하려면 콤마(,)로 연결", label="고정 체인"
                    )
                with gr.Column():
                    num_seqs = gr.Slider(
                        minimum=1, maximum=15, value=1, step=1, label="시퀀스 수"
                    )
                    sampling_temp = gr.Radio(
                        choices=["0.1", "0.15", "0.2", "0.25", "0.3"],
                        value="0.1",
                        label="샘플링 온도",
                    )
                with gr.Column():
                    model_name = gr.Dropdown(
                        choices=[
                            "vanilla—v_48_002",
                            "vanilla—v_48_010",
                            "vanilla—v_48_020",
                            "vanilla—v_48_030",
                            "soluble—v_48_010",
                            "soluble—v_48_020",
                        ],
                        label="Model",
                        value="vanilla—v_48_030",
                        visible=False,
                    )
                    backbone_noise = gr.Dropdown(
                        choices=["0", "0.02", "0.10", "0.20", "0.30"], label="Backbone noise", value="0", visible=False,
                    )
                with gr.Column():
                    homomer = gr.Checkbox(value=False, label="호모머(Homomer) 여부")
                    gr.Markdown(
                        "올바른 대칭적 결합을 위해서는 동종 사슬의 길이가 동일해야함"
                    )
                with gr.Column():
                    omit_AAs = gr.Textbox(
                        placeholder="Specify omitted amino acids ", label="Omitted amino acids", visible=False
                    )
                atomsel = gr.Textbox(
                    placeholder="Specify atom selection ", label="Fixed positions", visible=False
                )

        btn = gr.Button("실행", elem_id="click_button")
        
        gr.Markdown("# Output")

        with gr.Tabs():
            with gr.TabItem("디자인 시퀀스"):
                out = gr.Textbox(label="Status")

            with gr.TabItem("아미노산 결합확률 + T 조정 확률"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("아미노산 결합확률")
                        plot = gr.Plot()
                        all_log_probs = gr.File(visible=False)
      
                    with gr.Column():
                        gr.Markdown("T 조정 확률")
                        plot_tadjusted = gr.Plot()
                        all_probs = gr.File(visible=False)

            with gr.TabItem("구조 유효성 w/ AF2"):
                
                with gr.Row():
                    with gr.Row():
                        chosen_seq = gr.Dropdown(
                            choices=[],
                            label="검증을 위한 시퀀스를 선택하세요",
                            visible=False,
                        )
                        num_recycles = gr.Dropdown(
                            choices=["0", "1", "3", "5"], value="3", label="반복 횟수"
                        )
                    btnAF = gr.Button("모든 시퀀스에 대해 AlphaFold 실행", elem_id="click_button")

                with gr.Row():
                    with gr.Column(scale=5):
                        mol = gr.HTML()
                    with gr.Column(scale=4):
                        gr.Markdown("## Metrics")
                        p = {
                            0: {
                                "Seq": "NA",
                                "RMSD": "NA",
                                "Score": "NA",
                                "Recovery": "NA",
                                "Mean pLDDT": "NA",
                            }
                        }
                        placeholder = pd.DataFrame.from_dict(p, orient="index")
                        results = gr.Dataframe(
                            placeholder,
                            interactive=False,
                            row_count=(1, "dynamic"),
                            headers=["Seq", "RMSD", "Score", "Recovery", "Mean pLDDT"],
                        )
                        plotAF_plddt = gr.Plot(label="pLDDT")
                        # remove maxh80 class from css
                        plotAF_pae = gr.Gallery(label="PAE plots")  # gr.Plot(label="PAE")
        tempFile = gr.State()
        selectedResidues = gr.State()
        seq_dict = gr.State()
        btn.click(
            fn=update,
            inputs=[
                inp,
                file,
                designed_chain,
                fixed_chain,
                homomer,
                num_seqs,
                sampling_temp,
                model_name,
                backbone_noise,
                omit_AAs,
                atomsel,
            ],
            outputs=[
                out,
                plot,
                plot_tadjusted,
                all_log_probs,
                all_probs,
                tempFile,
                chosen_seq,
                selectedResidues,
                seq_dict,
            ],
        )
        btnAF.click(
            fn=update_AF,
            inputs=[seq_dict, tempFile, num_recycles, selectedResidues],
            outputs=[mol, plotAF_plddt, plotAF_pae, results],
        )
        #examples.click(fn=set_examples, inputs=examples, outputs=examples._components)

    ray.init(runtime_env={"working_dir": "./ProteinMPNN/af_backprop"})

    # proteinMPNN.launch(share=True)
