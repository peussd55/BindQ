import os
import subprocess
from Bio.PDB import PDBParser, Superimposer
import json
import shutil
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import tempfile
import ray
import sys
import nglview as nv
import py3Dmol

HOST_WORKSPACE_DIR = os.path.abspath('/home/eps/prj_envs/Gradio')
DOCKER_WORKSPACE_DIR = os.path.abspath('/workspace')

HOST_INPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'input')
DOCKER_INPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'input')

HOST_RFDIFFUSION_OUTPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'rfdiffusion_output')
DOCKER_RFDIFFUSION_OUTPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'rfdiffusion_output')

HOST_PROTEINMPNN_OUTPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'proteinmpnn_output')
DOCKER_PROTEINMPNN_OUTPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'proteinmpnn_output')

if f"{HOST_WORKSPACE_DIR}/af_backprop" not in sys.path:
    sys.path.append(f"{HOST_WORKSPACE_DIR}/af_backprop")

input_files = [f for f in os.listdir(HOST_INPUT_DIR) if f.endswith('.pdb')]
if not input_files:
    print('PDB files does not exists')
    exit(1)
#input_pdb_name = input_files[1].replace('.pdb', '')
try:
    input_pdb_name = input_files[1].replace('.pdb', '')
except IndexError:
    print("Error: Not enough PDB files in the input directory")


def save_uploaded_file(uploaded_file, flag):
    """
    Gradio에서 업로드한 파일을 HOST_INPUT_DIR에 저장
    :param uploaded_file: gr.File 컴포넌트로 업로드된 파일 (단일 파일)
    :return: 저장 성공 시 저장된 파일 경로 메시지, 실패 시 에러 메시지.
    """
    if not uploaded_file:
        return "파일이 업로드되지 않았습니다."

    file_name = os.path.basename(uploaded_file.name)
    # 파일 저장 경로
    if flag:
        target_path = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, file_name)
    else:
        target_path = os.path.join(HOST_INPUT_DIR, file_name)

    try:
        shutil.copy(uploaded_file.name, target_path)
        return f"✅ 업로드 완료: {target_path}", file_name
    except Exception as e:
        return f"❌ 업로드 중 에러 발생: {e}"


def process_inputs(name_output, contigs, hotspot_res, iterations, num_designs):
    """ Gradio 입력값을 JSON 형식의 override flags로 변환하여 run_inference()에 전달 """
    override_flags = {}

    if contigs and not (contigs.startswith('[') and contigs.endswith(']')):
        contigs = '[' + contigs + ']'

    if hotspot_res and not (hotspot_res.startswith('[') and hotspot_res.endswith(']')):
        hotspot_res = '[' + hotspot_res + ']'

    if name_output:
        if name_output.endswith(".pdb"):
            override_flags["inference.input_pdb"] = os.path.join(DOCKER_INPUT_DIR, name_output)
            override_flags["inference.output_prefix"] = os.path.join(
                DOCKER_RFDIFFUSION_OUTPUT_DIR, name_output.replace(".pdb", ""))
        else:
            override_flags["inference.output_prefix"] = os.path.join(
                DOCKER_RFDIFFUSION_OUTPUT_DIR, name_output)
    if contigs:
        override_flags["contigmap.contigs"] = contigs
    if hotspot_res:
        override_flags["ppi.hotspot_res"] = hotspot_res
    if iterations:
        override_flags["diffuser.T"] = iterations
    if num_designs:
        override_flags["inference.num_designs"] = int(num_designs)
    print('son.dumps(override_flags):', json.dumps(override_flags))
    return run_inference_RFdiffusion(json.dumps(override_flags))


def calculate_rmsd(file1, file2):
    parser = PDBParser(QUIET=True)

    structure1 = parser.get_structure("structure1", file1)
    structure2 = parser.get_structure("structure2", file2)

    # Cα 원자만 선택
    atoms1 = [atom for atom in structure1.get_atoms() if atom.get_name() == "CA"]
    atoms2 = [atom for atom in structure2.get_atoms() if atom.get_name() == "CA"]
    # # 첫 번째 모델과 첫 번째 체인 선택
    # atoms1 = list(structure1.get_atoms())
    # atoms2 = list(structure2.get_atoms())

    # 두 구조의 원자 수 맞추기 (길이가 다르면 최소 개수만큼 사용)
    min_length = min(len(atoms1), len(atoms2))
    atoms1 = atoms1[:min_length]
    atoms2 = atoms2[:min_length]

    # Superimposer 사용하여 RMSD 계산
    superimposer = Superimposer()
    superimposer.set_atoms(atoms1, atoms2)
    rmsd = superimposer.rms

    return rmsd


def calc_test(file1, file2):
    import MDAnalysis as mda
    from MDAnalysis.analysis import align
    # PDB 파일 로드
    ref = mda.Universe(file1)  # 기준 구조 (예: 실험 구조)
    mobile = mda.Universe(file2)  # 이동 구조 (예: AlphaFold 예측 구조)
    # C-alpha 원자 선택
    ref_calphas = ref.select_atoms("protein and name CA")
    mobile_calphas = mobile.select_atoms("protein and name CA")
    # 구조 정렬 (기준 구조에 mobile 구조를 정렬)
    min_atoms = min(len(mobile_calphas), len(ref_calphas))
    mobile_calphas = mobile_calphas[:min_atoms]
    ref_calphas = ref_calphas[:min_atoms]
    R = align.alignto(mobile_calphas, ref_calphas)
    # RMSD 계산 (정렬된 구조와 기준 구조 간의 RMSD)
    rmsd_val = np.sqrt(np.sum((mobile_calphas.positions - ref_calphas.positions)**2) / mobile_calphas.n_atoms)

    return rmsd_val


def extract_plddt_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    plddt_scores = []
    for atom in structure.get_atoms():
        if atom.get_name() == "CA":  # C-alpha 원자 선택
            plddt_scores.append(atom.get_bfactor())

    return np.array(plddt_scores)


def calculate_plddt_difference(pdb1, pdb2):
    plddt1 = extract_plddt_from_pdb(pdb1)
    plddt2 = extract_plddt_from_pdb(pdb2)

    # 길이가 다를 경우 최소 길이만큼 맞추기
    min_length = min(len(plddt1), len(plddt2))
    plddt1, plddt2 = plddt1[:min_length], plddt2[:min_length]

    return np.abs(plddt1 - plddt2)  # 차이 계산


def plot_plddt(pdb_file1, pdb_file2):
    plddt1 = extract_plddt_from_pdb(pdb_file1.name)
    plddt2 = extract_plddt_from_pdb(pdb_file2.name)

    # PLDDT Plot
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(plddt1, color="blue", label="Structure 1")
    ax1.plot(plddt2, color="red", linestyle="dashed", label="Structure 2")
    ax1.set_ylim(0, 100)
    ax1.set_title("PLDDT Score Comparison")
    ax1.set_xlabel("Residue Index")
    ax1.set_ylabel("PLDDT Score")
    ax1.legend()

    # PLDDT Difference Plot
    plddt_diff = calculate_plddt_difference(pdb_file1.name, pdb_file2.name)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.plot(plddt_diff, color="purple")
    ax2.set_ylim(0, 50)  # 차이값 범위 제한
    ax2.set_title("PLDDT Difference Between Two Structures")
    ax2.set_xlabel("Residue Index")
    ax2.set_ylabel("PLDDT Difference")

    return fig1, fig2


def visualize_pdbs(pdb_file1, pdb_file2):
    # PDB 파일 로드
    with open(pdb_file1.name, "r") as f:
        pdb_data1 = f.read()
    with open(pdb_file2.name, "r") as f:
        pdb_data2 = f.read()

    # HTML 코드 생성 - 3Dmol.js 라이브러리와 뷰어 초기화 코드 포함
    html_code = f"""
    <div style="height: 600px; width: 800px; position: relative;" class='viewer_3Dmoljs' data-href='{pdb_file1.name}' data-backgroundcolor='0xffffff'
        data-style='cartoon' data-ui='true'></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol.ui-min.js"></script>
    <script>
        $(document).ready(function() {{
            let viewer = $3Dmol.createViewer($(".viewer_3Dmoljs"), {{backgroundColor: "white"}});
            viewer.addModel(`{pdb_data1}`, "pdb");
            viewer.setStyle({{}}, {{cartoon: {{color: "blue"}}}});
            viewer.addModel(`{pdb_data2}`, "pdb");
            viewer.setStyle({{"model": 1}}, {{cartoon: {{color: "red"}}}});
            viewer.zoomTo();
            viewer.render();
        }});
    </script>
    """
    return html_code


# input_pdb_path = os.path.join(HOST_INPUT_DIR, input_files[0])
# rfdiffusion_output_pdb = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, f'rf_{input_pdb_name}')

# # Run RFdiffusion Container
# print(f'Run RFdiffusion Container - {input_pdb_path}')
# subprocess.run([
#     'docker', 'run', '--rm', '-it', '--gpus', 'all',
#     '-v', f'{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}', 'rfdiffusion:latest',
#     'python', 'scripts/run_inference.py', f'inference.output_prefix={DOCKER_RFDIFFUSION_OUTPUT_DIR}/{input_pdb_name}', 'inference.model_directory_path=models/',
#     f'inference.input_pdb={DOCKER_INPUT_DIR}/{input_files[0]}', 'inference.num_designs=1', 'contigmap.contigs=[10-40/A163-181/10-40]'
# ])

# # Results Rfdiffusion
# rfd_out_files = [f for f in os.listdir(HOST_RFDIFFUSION_OUTPUT_DIR) if f.endswith('.pdb')]
# if not rfd_out_files:
#     print('❌ RFdiffusion output PDB files does not exists')
#     exit(1)


def run_inference_RFdiffusion(override_flags_json: str) -> str:
    """
    Gradio를 통해 입력받은 JSON 형식의 flag override를 파싱하여,
    기본 base.yaml 설정은 컨테이너 내부에서 사용하고,
    입력받은 값만 command line 인자로 전달합니다.
    """
    # 사용자가 입력한 override 값이 있을 경우 파싱 (없으면 빈 dict)
    override_flags = {}
    if override_flags_json.strip():
        try:
            override_flags = json.loads(override_flags_json)
            if not isinstance(override_flags, dict):
                return "JSON 입력은 key-value 쌍의 객체여야 합니다."
        except Exception as e:
            return f"JSON 파싱 에러: {e}"
    
    ### 추가됨 ###
    print("override_flags:", override_flags)
    print("num수:", override_flags["inference.num_designs"])
    print("id명:", (override_flags["inference.input_pdb"]).split('/')[-1].split('.')[0])
    num_desgin = override_flags["inference.num_designs"]
    name_output = (override_flags["inference.input_pdb"]).split('/')[-1].split('.')[0]
    contigs = ""
    hotspots = ""
    if "contigmap.contigs" in override_flags:
        contigs = override_flags["contigmap.contigs"]
    if "ppi.hotspot_res" in override_flags:
        hotspots = override_flags["ppi.hotspot_res"]

    # override flag들을 "key=value" 형태의 문자열로 변환
    flag_args = [f"{key}={value}" for key, value in override_flags.items()]



    # Docker run 명령어 구성.
    # 컨테이너 내부에는 base.yaml 파일이 있어 기본 설정을 로드하므로,
    # 추가로 전달되는 인자만 override 됩니다.
    docker_cmd = [
        'docker', 'run', '--rm', '-it', '--gpus', 'all',
        #'--network', 'host', # 호스트 네트워크 모드 사용
        '-v', f'{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}',
        'rfdiffusion:latest',
        'python', 'scripts/run_inference.py',
    ] + flag_args

    print("실행할 명령어:", " ".join(docker_cmd))
    try:
        subprocess.run(docker_cmd, check=True)

        ### 추가됨 ###
        from function import render_pdbs_by_prefix
        return render_pdbs_by_prefix(name_output, num_desgin, contigs, hotspots) 
        
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

    #return "실행 완료"

# # ProteinMPNN Input
# proteinmpnn_input_pdb_name = rfd_out_files[0].replace('.pdb', '')
# proteinmpnn_input_pdb_path = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, f'{rfd_out_files[0]}')

# # Run ProteinMPNN Container
# print(f"🚀 ProteinMPNN 실행: 입력 파일 - {proteinmpnn_input_pdb_name}")
# subprocess.run([
#     'docker', 'run', '--rm', '-it', '--gpus', 'all',
#     "-v", f"{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}",
#     'protein_mpnn:latest',
#     "python", "protein_mpnn_run.py",
#     '--pdb_path', f'{DOCKER_RFDIFFUSION_OUTPUT_DIR}/{rfd_out_files[0]}',
#     '--out_folder', f'{DOCKER_PROTEINMPNN_OUTPUT_DIR}',
#     '--num_seq_per_target', '5',
#     '--batch_size', '1',
#     '--seed', '32'
# ])
# # Results ProteinMPNN
# proteinmpnn_output = [f for f in os.listdir(f'{HOST_PROTEINMPNN_OUTPUT_DIR}/seqs') if f.endswith('.fa')]
# if not proteinmpnn_output:
#     print("❌ ProteinMPNN 실행 실패: FASTA 출력 없음.")
#     exit(1)

# print("✅ All Process Complete!")


def run_inference_ProteinMPNN(target_file_1,
                              target_file_2,
                              name_output,
                              num_seq_per_target,
                              sampling_temp,
                              model_name,
                              backbone_noise
                              ):

    # if designed_chain == "":
    #     designed_chain_list = []
    # else:
    #     designed_chain_list = re.sub("[^A-Za-z]+", ",", designed_chain).split(",")

    # if fixed_chain == "":
    #     fixed_chain_list = []
    # else:
    #     fixed_chain_list = re.sub("[^A-Za-z]+", ",", fixed_chain).split(",")

    # chain_list = list(set(designed_chain_list + fixed_chain_list))

    docker_cmd = [
        'docker', 'run', '--rm', '-it', '--gpus', 'all',
        '-v', f'{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}',
        'protein_mpnn:latest'
    ]

    # # protein_mpnn_run.py 실행
    docker_cmd.append(
        f'python protein_mpnn_run.py '
        f'--pdb_path {DOCKER_RFDIFFUSION_OUTPUT_DIR}/{name_output} '
        f'--out_folder {DOCKER_PROTEINMPNN_OUTPUT_DIR} '
        f'--seed 32 --batch_size 1 --save_probs 1 '
    )
    
    if num_seq_per_target:
        docker_cmd.append(f'--num_seq_per_target {num_seq_per_target} ')
    # if sampling_temp:
    #     docker_cmd.append(f'--sampling_temp {sampling_temp} ')
    # if model_name:
    #     _, model_name = model_name.split("—")
    #     docker_cmd.append(f'--model_name {model_name} ')
    # if backbone_noise:
    #     docker_cmd.append(f'--backbone_noise {backbone_noise} ')

    # try:
    #     subprocess.run(" ".join(docker_cmd), shell=True, check=True)
    # except subprocess.CalledProcessError as e:
    #     return f"Docker 실행 중 에러 발생: {e}"

    alphabet = "ACDEFGHIKLMNPQRSTVWYX"

    name_output = name_output.replace('.pdb', '')
    data = np.load(f'{HOST_PROTEINMPNN_OUTPUT_DIR}/probs/{name_output}.npz')
    all_log_probs = data['log_probs']
    all_probs = data['probs']
    np.savetxt(
        f'{HOST_PROTEINMPNN_OUTPUT_DIR}/probs/all_log_probs_{name_output}.csv',
        np.exp(all_log_probs).mean(0).T,
        delimiter=','
    )
    np.savetxt(
        f'{HOST_PROTEINMPNN_OUTPUT_DIR}/probs/all_probs_{name_output}.csv',
        np.exp(all_probs).mean(0).T,
        delimiter=','
    )

    fig = px.imshow(
        np.exp(all_log_probs).mean(0).T,
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )
    fig.update_xaxes(side="top")

    fig_tadjusted = px.imshow(
        all_probs.mean(0).T,
        labels=dict(x="positions", y="amino acids", color="probability"),
        y=list(alphabet),
        template="simple_white",
    )
    # rmsd_out = calculate_rmsd(target_file_1, target_file_2)
    rmsd_out = calculate_rmsd(target_file_1, target_file_2)
    fig1, fig2 = plot_plddt(target_file_1, target_file_2)
    vis_out = visualize_pdbs(target_file_1, target_file_2)
    
    rmsd_out = rmsd_out if rmsd_out is not None else "No Data"
    fig1 = fig1 if fig1 is not None else gr.Plot()
    fig2 = fig2 if fig2 is not None else gr.Plot()
    vis_out = vis_out if vis_out is not None else "No Visualization"
    fig = fig if fig is not None else gr.Plot()
    fig_tadjusted = fig_tadjusted if fig_tadjusted is not None else gr.Plot()
    
    return (
        rmsd_out,
        fig1,
        fig2,
        vis_out,
        '',
        fig,
        fig_tadjusted,
        gr.File(
            value=f'{HOST_PROTEINMPNN_OUTPUT_DIR}/probs/all_log_probs_{name_output}.csv',
            visible=True
        ),
        gr.File(
            value=f'{HOST_PROTEINMPNN_OUTPUT_DIR}/probs/all_probs_{name_output}.csv',
            visible=True
        )
    )
