import os
import subprocess
import yaml
import json
import shutil

HOST_WORKSPACE_DIR = os.path.abspath('/mnt/c/Users/user/Desktop/FinalProject/Gradio')
DOCKER_WORKSPACE_DIR = os.path.abspath('/workspace')

HOST_INPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'input')
DOCKER_INPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'input')

HOST_RFDIFFUSION_OUTPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'rfdiffusion_output')
DOCKER_RFDIFFUSION_OUTPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'rfdiffusion_output')

HOST_PROTEINMPNN_OUTPUT_DIR = os.path.join(HOST_WORKSPACE_DIR, 'proteinmpnn_output')
DOCKER_PROTEINMPNN_OUTPUT_DIR = os.path.join(DOCKER_WORKSPACE_DIR, 'proteinmpnn_output')

input_files = [f for f in os.listdir(HOST_INPUT_DIR) if f.endswith('.pdb')]
if not input_files:
    print('PDB files does not exists')
    exit(1)
input_pdb_name = input_files[1].replace('.pdb', '')


def save_uploaded_file(uploaded_file):
    """
    Gradio에서 업로드한 파일을 HOST_INPUT_DIR에 저장
    :param uploaded_file: gr.File 컴포넌트로 업로드된 파일 (단일 파일)
    :return: 저장 성공 시 저장된 파일 경로 메시지, 실패 시 에러 메시지.
    """
    if not uploaded_file:
        return "파일이 업로드되지 않았습니다."

    file_name = os.path.basename(uploaded_file.name)
    # 파일 저장 경로
    target_path = os.path.join(HOST_INPUT_DIR, file_name)

    try:
        shutil.copy(uploaded_file.name, target_path)
        return f"✅ 업로드 완료: {target_path}", file_name
    except Exception as e:
        return f"❌ 업로드 중 에러 발생: {e}"


def process_inputs(name_output, contigs, hotspot_res, iterations, num_designs):
    """ Gradio 입력값을 JSON 형식의 override flags로 변환하여 run_inference()에 전달 """
    override_flags = {}

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

    return run_inference_RFdiffusion(json.dumps(override_flags))


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

    # override flag들을 "key=value" 형태의 문자열로 변환
    flag_args = [f"{key}={value}" for key, value in override_flags.items()]

    # Docker run 명령어 구성.
    # 컨테이너 내부에는 base.yaml 파일이 있어 기본 설정을 로드하므로,
    # 추가로 전달되는 인자만 override 됩니다.
    docker_cmd = [
        'docker', 'run', '--rm', '-it', '--gpus', 'all',
        '-v', f'{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}',
        'rfdiffusion:latest',
        'python', 'scripts/run_inference.py',
    ] + flag_args

    print("실행할 명령어:", " ".join(docker_cmd))
    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Error: {e}"

    return "실행 완료"

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


def run_inference_ProteinMPNN(num_seq_per_target, sampling_temp, model_name, backbone_noise):

    docker_cmd = [
        'docker', 'run', '--rm', '-it', '--gpus', 'all',
        '-v', f'{HOST_WORKSPACE_DIR}:{DOCKER_WORKSPACE_DIR}',
        'protein_mpnn:latest',
        'python', 'protein_mpnn_run.py',
        '--out_folder', f'{DOCKER_PROTEINMPNN_OUTPUT_DIR}',
        '--seed', '32',
        '--batch_size', '1'
    ]

    if num_seq_per_target:
        docker_cmd.extend(["--num_seq_per_target", str(num_seq_per_target)])
    if sampling_temp:
        docker_cmd.extend(["--sampling_temp", sampling_temp])
    if model_name:
        docker_cmd.extend(["--model_name", model_name])
    if backbone_noise:
        docker_cmd.extend(["--backbone_noise", backbone_noise])

    try:
        subprocess.run(docker_cmd, check=True)
    except subprocess.CalledProcessError as e:
        return f"Docker 실행 중 에러 발생: {e}"
    return
