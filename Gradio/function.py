import requests
import glob
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB import PDBIO
import os
import zipfile
import pandas as pd
import subprocess
import shutil
import json
import logging
import re
import gradio as gr
import numpy as np
from run_pipeline import HOST_INPUT_DIR, HOST_RFDIFFUSION_OUTPUT_DIR, DOCKER_RFDIFFUSION_OUTPUT_DIR, HOST_WORKSPACE_DIR
import plotly.graph_objects as go
import stat

tabs = ["HOME", "구조기반 결합부위 예측", "De novo 단백질 백본 설계", "후보물질 서열 생성", "도킹 및 결합도 검증"] 
def show_tab_content(tab_index):
    return [gr.update(visible=(i == tab_index)) for i in range(len(tabs))]
def toggle_image(visible):
    return (
        gr.update(visible=not visible),
        gr.update(visible=not visible),
        not visible
    )
def toggle_image_0(visible):
    return (
        gr.update(visible=not visible),
        not visible
    )
def chat_response(message, history):
    # Simple echo bot for demonstration
    # Replace with your actual chatbot logic
    return message

def display_file_content(file):
    # 파일 내용을 읽어와 InputBox에 표시하는 함수
    if file is not None:
        try:
            with open(file.name, "r", encoding="utf-8") as f:  # 파일 경로를 열어서 읽음
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    return ""  # 파일이 없으면 빈 문자열 반환

def handle_upload(pdb_code, pdb_file, flag):
    """입력된 값이 PDB 코드인지 파일인지 판별 후 적절한 함수 실행"""
    if pdb_code is None or pdb_code.strip() == "":
        return save_uploaded_file(pdb_file, flag)
    else:
        return get_pdb_file(pdb_code, flag)

def get_pdb_file(pdb_code, flag):
    pdb_code = pdb_code.strip().upper()

    # 복합체(ID 4자리. ex : 4hhb, 1o91, 5tpn ...) / 단량체(ID 6자리. ex : P68871, P00533 ...) filename 전처리
    if len(pdb_code) == 4:
        pdb_filename = f"{pdb_code}.pdb"
    elif len(pdb_code) >= 6:
        pdb_code = pdb_code.upper() # 대문자 여야함.
        pdb_filename = f"AF-{pdb_code}-F1-model_v4.pdb"

    if flag:
        target_path = HOST_RFDIFFUSION_OUTPUT_DIR
    else:
        target_path = HOST_INPUT_DIR

    pdb_path = os.path.join(target_path, pdb_filename)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if not os.path.exists(pdb_path):
        if len(pdb_code) == 4:  # PDB ID는 항상 4자리
            url = f"https://files.rcsb.org/view/{pdb_filename}"
        elif len(pdb_code) >= 6:  # UniProt ID는 일반적으로 6자리 이상
            url = f"https://alphafold.ebi.ac.uk/files/{pdb_filename}"

        command = ["wget", "-qnc", url, "-O", pdb_path]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return f"❌ 다운로드 실패: {pdb_code}", ""

    return f"✅ 다운로드 완료: {pdb_code}", pdb_filename

def get_pdb_file2(pdb_code, flag):
    pdb_code = pdb_code.strip().upper()

    # 복합체(ID 4자리. ex : 4hhb, 1o91, 5tpn ...) / 단량체(ID 6자리. ex : P68871, P00533 ...) filename 전처리
    if len(pdb_code) == 4:
        pdb_filename = f"{pdb_code}.pdb"
    elif len(pdb_code) >= 6:
        pdb_code = pdb_code.upper() # 대문자 여야함.
        pdb_filename = f"AF-{pdb_code}-F1-model_v4.pdb"

    if flag:
        target_path = HOST_RFDIFFUSION_OUTPUT_DIR
    else:
        target_path = HOST_INPUT_DIR

    if len(pdb_code) == 4:
        pdb_path = os.path.join(target_path, pdb_filename)
    elif len(pdb_code) >= 6:
        # 여기서 파일 이름을 pdb_code로 변경
        pdb_path = os.path.join(target_path, f"{pdb_code}.pdb")
    
    print('pdb_path:', pdb_path)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    if not os.path.exists(pdb_path):
        if len(pdb_code) == 4:  # PDB ID는 항상 4자리
            url = f"https://files.rcsb.org/view/{pdb_filename}"
        elif len(pdb_code) >= 6:  # UniProt ID는 일반적으로 6자리 이상
            url = f"https://alphafold.ebi.ac.uk/files/{pdb_filename}"

        command = ["wget", "-qnc", url, "-O", pdb_path]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return f"{pdb_code}"

    return f"{pdb_code}"


def plot_pLDDT(plddt_scores):
    """AlphaFold2 pLDDT 스코어 시각화"""
    x = np.arange(len(plddt_scores))
    fig = go.Figure(data=[go.Scatter(x=x, y=plddt_scores, mode='lines+markers')])
    fig.update_layout(title="AlphaFold pLDDT Score", xaxis_title="Residue Index", yaxis_title="pLDDT")
    return fig
    
def render_pdbs_by_prefix(name_output, num_desgin, contigs, hotspots):
    # PDB ID 추출
    print("name_output :", name_output)
    if not name_output.startswith('AF'):
        name_output = name_output.lower()
    base_name = os.path.splitext(name_output)[0]
    base_name = base_name.upper()
    print("base_name :", base_name)

    #하위디렉토리 생성
    contigs = (contigs.strip()).replace("/","").replace(",", "").replace("]", "").replace("[", "")
    hotspots = (hotspots.strip()).replace("/","").replace(",", "").replace("]", "").replace("[", "")
    DIR_SUB_BASENAME = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, base_name)
    DIR_SUB_CONTIGS = os.path.join(DIR_SUB_BASENAME, contigs)
    DIR_SUB_CONTIGS_HOTSPOTS = os.path.join(DIR_SUB_CONTIGS, hotspots)
    os.makedirs(DIR_SUB_CONTIGS_HOTSPOTS, exist_ok=True)  # 디렉토리가 없으면 생성
    print("DIR_SUB_CONTIGS_HOTSPOTS:",DIR_SUB_CONTIGS_HOTSPOTS)

    try:
        # pdb파일 경로 추출
        file_pattern = f"{HOST_RFDIFFUSION_OUTPUT_DIR}/{base_name}_*.pdb"
        #pdb_files = sorted(glob.glob(file_pattern))  # 파일 이름 정렬
        pdb_files = sorted(glob.glob(file_pattern))[:num_desgin]  # 파일 이름 정렬 후 num_design 개수만큼 슬라이싱
        print("pdb_files::",pdb_files)

        # trb파일 경로 추출
        file_pattern_trb = f"{HOST_RFDIFFUSION_OUTPUT_DIR}/{base_name}_*.trb"
        #trb_files = sorted(glob.glob(file_pattern_trb))  # 파일 이름 정렬
        trb_files = sorted(glob.glob(file_pattern_trb))[:num_desgin]  # 파일 이름 정렬 후 num_design 개수만큼 슬라이싱
        print("trb_files::",trb_files)

        # traj디렉토리 경로 추출
        dir_traj = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, "traj")
        print("dir_traj::",dir_traj)

        if not pdb_files:
            return f"<p>No files found with prefix '{base_name}'</p>"

        # 여러 iframe을 한 행에 3개씩 배치하도록 HTML 생성
        # html_content = """
        # <div style="display: flex; flex-wrap: wrap; gap: 10px; justify-content: space-between;">
        # """
        html_content = """
            <style>
            .pdb-container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                width: 100%;
            }
            .pdb-item {
                text-align: center;
                min-width: 200px;
            }
            </style>
            <div class="pdb-container">
        """

        # trb 리스트의 각 파일 경로를 반복하면서 파일 삭제 : 필요시 주석
        for trb_path in trb_files:
            # 파일이 존재하는지 확인
            if os.path.exists(trb_path):
                try:
                    os.remove(trb_path)
                    print(f"파일 삭제 완료: {trb_path}")
                except PermissionError:
                    print(f"권한 오류: {trb_path} 파일을 삭제할 수 없습니다.")
                except Exception as e:
                    print(f"오류 발생: {trb_path} 삭제 중 {e}")
            else:
                print(f"파일이 존재하지 않습니다: {trb_path}")

        # dir_traj 디렉토리 하위 모두 삭제(_pX0_traj.pdb 파일 들어있음) : 필요시 주석. 권한문제로 작동X
        if os.path.exists(dir_traj):
            try:
            # 디렉토리 트리를 순회하며 권한 변경
                for root, dirs, files in os.walk(dir_traj, topdown=False):
            # 모든 파일의 권한 변경
                    for file in [os.path.join(root, f) for f in files]:
                        try:
                            os.chmod(file, 0o755)
                        except Exception as e:
                            print(f"파일 권한 변경 중 오류 발생: {file} - {e}")

                    # 모든 디렉토리의 권한 변경
                    for dir in [os.path.join(root, d) for d in dirs]:
                        try:
                            os.chmod(dir, 0o755)
                        except Exception as e:
                            print(f"디렉토리 권한 변경 중 오류 발생: {dir} - {e}")
                
                # 루트 디렉토리(dir_traj)의 권한도 변경
                os.chmod(dir_traj, 0o755)
                print("dir_traj 디렉토리와 모든 하위 요소의 권한이 755로 변경되었습니다.")
                
                # 디렉토리와 모든 내용 삭제
                shutil.rmtree(dir_traj)
                print("dir_traj 디렉토리와 모든 내용이 성공적으로 삭제되었습니다.")
            except Exception as e:
                print(f"작업 중 오류가 발생했습니다: {e}")
            else:
                print("dir_traj 디렉토리가 존재하지 않습니다.")


        # 하위디렉토리에서 파일처리후 렌더링
        for file_path in pdb_files:
            print("filepath::", file_path)
            try:
                with open(file_path, "r") as pdb_file:
                    pdb_data = pdb_file.read()
                
                # PDB 파일 이름 추출
                file_name = os.path.basename(file_path)

                # 하위디렉토리로 이동
                destination_path = os.path.join(DIR_SUB_CONTIGS_HOTSPOTS, file_name)
                shutil.move(file_path, destination_path)

                # PDB 파일 다운로드 경로
                #file_down_path = file_path.replace("/home/eps/prj_envs/Gradio", "")
                #print("file_down_path::", file_down_path)
                file_down_path = destination_path.replace(HOST_WORKSPACE_DIR, "")
                print("file_down_path::", file_down_path)

                # 각 PDB 데이터를 iframe으로 렌더링 (정적파일 엔드포인트추가안되어서 다운안됨.)
                iframe_html = generate_3dmol_html(pdb_data)
                # html_content += f"""
                # <div style="flex: 0 0 calc(33.33% - 10px); box-sizing: border-box; min-width: 200px; text-align: center;">
                #     {iframe_html}
                #     <p style="margin-top: 5px; font-size: 14px; color: #555;">
                #         <a href="{file_path}" download="{file_name}" style="text-decoration: none; color: #007BFF;">
                #             {file_name}
                #         </a>
                #     </p>
                # </div>
                # """
                html_content += f"""
                <div class="pdb-item">
                    {iframe_html}
                    <p style="margin-top: 5px; font-size: 24px; color: #555;">
                        <a href="{file_down_path}" download="{file_name}" style="text-decoration: none; color: white;">
                            {file_name}
                        </a>
                    </p>
                </div>
                """
            except Exception as e:
                html_content += f"<p>Error loading PDB file '{file_path}': {str(e)}</p>"
        
        html_content += "</div>"
        return html_content

    except Exception as e:
        return f"<p>Error processing files: {str(e)}</p>"
    

def render_pdbs_by_prefix_prodigy(output_path, num_size, dir_yn=1):
    print("output_path:", output_path)
    try:
        # 디렉토리 내 모든 파일 목록 가져오기
        all_files = []
        if dir_yn == 0:
            for file in os.listdir(output_path):
                file_path = os.path.join(output_path, file)
                if os.path.isfile(file_path):  # 디렉토리가 아닌 파일만 추가
                    all_files.append(file_path)
        else:
            all_files.append(output_path)
        
        # 파일 정렬 후 num_size 개수만큼 슬라이싱
        pdb_files = sorted(all_files)[:num_size]
        
        if not pdb_files:
            return f"<p>No files found in directory '{output_path}'</p>"

        # HTML 컨테이너 시작
        html_content = """
            <style>
            .pdb-container {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 15px;
                width: 100%;
            }
            .pdb-item {
                text-align: center;
                min-width: 200px;
            }
            </style>
            <div class="pdb-container">
        """
        
        # 각 파일에 대해 처리
        for file_path in pdb_files:
            print(file_path)
            try:
                with open(file_path, "r") as pdb_file:
                    pdb_data = pdb_file.read()
                
                # 파일 이름 추출
                file_name = os.path.basename(file_path)
                file_name = file_name.replace("_prefix.pdb", "")
                
                # 3D 모델 HTML 생성
                iframe_html = generate_3dmol_html(pdb_data)
                html_content += f"""
                <div class="pdb-item">
                    {iframe_html}
                    <p style="margin-top: 5px; font-size: 24px; color: white;">
                        {file_name}
                    </p>
                </div>
                """
            except Exception as e:
                html_content += f"<p>Error loading file '{file_path}': {str(e)}</p>"

        html_content += "</div>"
        return html_content

    except Exception as e:
        return f"<p>Error processing directory: {str(e)}</p>"

    
# 3Dmol.js를 사용한 HTML 생성 함수
def generate_3dmol_html(pdb_data):
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <script src="https://3dmol.csb.pitt.edu/build/3Dmol-min.js"></script>
    </head>
    <body>
        <div style="border: 1px solid black; width: 100%; height: 400px; position: relative;" id="viewer"></div>
        <script>
            let viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
            viewer.addModel(`{pdb_data}`, "pdb");
            viewer.setStyle({{}}, {{cartoon: {{color: "spectrum"}}}});
            viewer.zoomTo();
            viewer.render();
        </script>
    </body>
    </html>
    """
    return f"<iframe srcdoc='{html_content}' width='100%' height='400px' frameborder='0' scrolling='no'></iframe>"

# PDB 파일 URL 생성 함수 (PDB ID 또는 UniProt ID)
def generate_pdb_url(input_id):
    input_id = input_id.strip().upper()  # 공백 제거 및 대문자 변환
    print("generate_pdb_url")
    # PDB ID인지 UniProt ID인지 확인
    if len(input_id) == 4:  # PDB ID는 항상 4자리 (복합체는 검색X)
        return f"https://files.rcsb.org/download/{input_id}.pdb", "PDB"
    elif len(input_id) >= 6:  # UniProt ID는 일반적으로 6자리 이상
        return f"https://alphafold.ebi.ac.uk/files/AF-{input_id}-F1-model_v4.pdb", "AlphaFold"
    else:
        return None, None

# 3Dmol.js HTML 생성 함수
def generate_molstar_html_scannet(input_id):
    print('generate_molstar_html_scannet_input_id:', input_id)
    input_id = input_id #upper()
    pdb_url, id_type = generate_pdb_url(input_id)
    
    # if not pdb_url:
    #     return "<p>입력된 ID가 유효하지 않습니다. PDB ID 또는 UniProt ID를 입력하세요.</p>"
    
    # # URL 유효성 확인
    # response = requests.head(pdb_url)
    # if response.status_code != 200:
    #     return f"<p>{id_type} 데이터베이스에서 '{input_id}'에 대한 구조를 찾을 수 없습니다.</p>"
    # print('pdb_url:', pdb_url)
    iframe_src = f"/static/scannet/results/{input_id}_pdb.html"
    print('iframe_src:', iframe_src)
    full_path = os.path.join(os.getcwd(), iframe_src.lstrip('/'))
    print('full_path:', full_path)

    if os.path.exists(full_path):
        return f"""
        <iframe
            id="molstar-frame"
            style="width: 100%; height: 900px; border: none;"
            src="{iframe_src}">
        </iframe>
        """ 
    else:
        return f"""
        <div>
            해당 단백질 정보가 존재하지 않습니다. "단백질 결합 부위 예측하기" 버튼을 눌러 결과를 생성하세요.
        </div>
        """

# PDB파일 다운로드 링크 생성 함수 (molstart viewer반환은 제거함함)
def generate_pdb_down_link(input_id):
    pdb_url, id_type = generate_pdb_url(input_id)
    print("pdb_url::", pdb_url)
    print("id_type::", id_type)
    pdb_url2 = f"""
    <p>{id_type} 데이터베이스에서 '{input_id}'에 대한 구조를 찾았습니다.</p>
    <a href="{pdb_url}" download="{input_id}.pdb" target="_blank">
        <button style="padding: 10px; background-color: blue; color: white; border: none; cursor: pointer;">
            PDB 파일 다운로드
        </button>
    </a>
    """

    if not pdb_url:
        return "<p>입력된 ID가 유효하지 않습니다. 다시 입력하세요.</p>", None
    
    # response = requests.head(pdb_url)
    # if response.status_code != 200:
    #     return f"<p>{id_type} 데이터베이스에서 '{input_id}'에 대한 구조를 찾을 수 없습니다.</p>", None
    
    # html_content = f"""
    # <iframe
    #     id="molstar-frame"
    #     style="width: 100%; height: 600px; border: none;"
    #     srcdoc='
    #     <!DOCTYPE html>
    #     <html lang="en">
    #         <head>
    #             <script src="/static/molstar.js"></script>
    #             <link rel="stylesheet" type="text/css" href="/static/molstar.css" />
    #         </head>
    #         <body>
    #             <div id="molstar-viewer" style="width: 100%; height: 600px;"></div>
    #             <script>
    #                 (async function() {{
    #                 const viewer = await molstar.Viewer.create("molstar-viewer", {{ layoutIsExpanded: true }});
    #                 await viewer.loadStructureFromUrl("{pdb_url}", "pdb");
    #                 }})();
    #             </script>
    #         </body>
    #     </html>'>
    # </iframe>
    # """
    return pdb_url2

# 데이터프레임 업데이트
def update_data_table(input_id):
    # input_id를 기반으로 CSV 파일 처리 로직 추가
    input_id = input_id.upper()
    #print('input_id:', input_id)

    try:
        # 예시: 특정 파일 경로와 열 이름을 사용하여 데이터프레임 생성
        file_path = f"./file/{input_id}/predictions_{input_id}.csv"  # input_id를 사용하여 파일 경로 생성
        print(file_path)
        sort_column = "Binding site probability"  # 정렬 기준 열 이름 (예시)
        sorted_df = load_and_sort_csv(file_path, sort_column)
        return sorted_df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})  # 에러 발생 시 에러 메시지를 데이터프레임으로 반환

FLASK_SERVER_URL = "http://172.30.65.76:5002/predict"
def send_to_flask_server(pdb_code, pdb_path):
    print("pdb_code::"+pdb_code)
    if not pdb_code.strip():
        return "ID를 입력하세요.", ''
    try:
        # 입력값을 대문자로 변환
        pdb_code = pdb_code.strip() #upper()
        #print('pdb_code :', pdb_code)

         # 파일 저장 디렉토리 설정
        output_dir = "file"
        os.makedirs(output_dir, exist_ok=True)  # 디렉토리가 없으면 생성

        # Flask 서버로 POST 요청 보내기
        response = requests.post(FLASK_SERVER_URL, json={"id": pdb_code, "pdb_path": pdb_path}, stream=True)  # stream=True로 설정
        file_path = ""
        # 응답 상태 코드 확인
        if response.status_code == 200:
            # ZIP 파일 저장 경로 설정
            zip_file_path = os.path.join(output_dir, f"{pdb_code}_predictions.zip")
            with open(zip_file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print(f"ZIP 파일이 저장되었습니다: {zip_file_path}")

            # ZIP 파일 압축 해제
            extract_dir = os.path.join(output_dir, f"{pdb_code}")
            os.makedirs(extract_dir, exist_ok=True)  # 압축 해제 디렉토리 생성

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)

            print(f"ZIP 파일이 압축 해제되었습니다: {extract_dir}")

            # 원본 ZIP 파일 삭제
            # os.remove(zip_file_path)
            # print(f"원본 ZIP 파일이 삭제되었습니다: {zip_file_path}")

            modify_pdb_file(pdb_code)
            pdb_url, id_type = generate_pdb_url(pdb_code)

            #return f"파일이 성공적으로 저장 및 압축 해제되었습니다: {extract_dir}", pdb_url
            #return pdb_url
            return f"{pdb_code}.pdb 파일생성 완료."
        else:
            # 에러 메시지 반환
            return f"Flask 서버 요청 실패! 상태 코드: {response.status_code}, 메시지: {response.json()}"

    except Exception as e:
        return f"Flask 서버 요청 중 오류 발생: {str(e)}"



# 디퓨전 목록에서 3d렌더링
def render_3d_molecules(name_output):
    # 압축 해제된 파일 경로 설정
    extract_to_path = './pdbs/generate'
    base_name = os.path.splitext(name_output)[0]
    # name_output_숫자.pdb 형식의 파일 검색
    pdb_files = glob.glob(os.path.join(extract_to_path, f"{base_name}_*.pdb"))

    print('base_name :', base_name)
    print('pdb_files :', pdb_files)

    # 3Dmol.js HTML 생성
    html_content = ""
    for i, pdb_file in enumerate(pdb_files):
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()

        viewer_id = f"viewer_{i}"
        html_content += f"""
        <div style="display: inline-block; margin: 10px;">
            <div id="{viewer_id}" style="width: 400px; height: 400px;"></div>
            <script>
                let viewer_{i} = $3Dmol.createViewer("{viewer_id}", {{ backgroundColor: "white" }});
                viewer_{i}.addModel(`{pdb_data}`, "pdb");
                viewer_{i}.setStyle({{}}, {{stick: {{}}, cartoon: {{color: "spectrum"}}}});
                viewer_{i}.zoomTo();
                viewer_{i}.render();
            </script>
        </div>
        """

    # 반환할 HTML
    return f"""
    <div style="display: flex; flex-wrap: wrap; justify-content: center;">
        {html_content}
    </div>
    """

# CSV 파일 읽기 및 데이터프레임 생성 함수
def load_and_sort_csv(file_path, sort_column):
    #print('file_path: ', file_path)
    try:
        # 파일 경로 유효성 확인
        if not os.path.exists(file_path):
            return pd.DataFrame()  # 빈 데이터프레임 반환 (공백)

        # CSV 파일 로드
        df = pd.read_csv(file_path)

        # 첫 번째 열 제거
        df = df.iloc[:, 1:]  # 또는 df.drop(df.columns[0], axis=1, inplace=False)

        # 특정 열 기준으로 내림차순 정렬
        sorted_df = df.sort_values(by=sort_column, ascending=False)
        return sorted_df
    except Exception as e:
        return str(e)
    
# cif to pdb
def convert_cif_to_pdb(cif_file, pdb_file):
    parser = MMCIFParser()
    structure = parser.get_structure("structure", cif_file)
    io = PDBIO()
    io.set_structure(structure)
    io.save(pdb_file)

# pdb.html 파일을 렌더링파일로 변환
def modify_pdb_file(id):
    # 경로 설정
    input_file_path = './static/scannet/results/pdb.html'
    pdb_file_path = f'./file/{id}/annotated_{id}.pdb'
    output_file_path = f'./static/scannet/results/{id}_pdb.html'

    try:
        # PDB 파일 읽기
        if not os.path.exists(pdb_file_path):
            return f"오류: PDB 파일 '{pdb_file_path}'이 존재하지 않습니다."
        
        with open(pdb_file_path, 'r', encoding='utf-8') as pdb_file:
            pdb_content = pdb_file.read()

        # 입력 HTML 파일 읽기
        if not os.path.exists(input_file_path):
            return f"오류: 입력 HTML 파일 '{input_file_path}'이 존재하지 않습니다."
        
        with open(input_file_path, 'r', encoding='utf-8') as html_file:
            html_content = html_file.read()

        # HTML 내용 수정
        replacement = f'var stringContainingTheWholePdbFile = `{pdb_content}`;'
        modified_content = html_content.replace('var stringContainingTheWholePdbFile = ``;', replacement)

        # 수정된 내용을 새로운 파일에 저장
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write(modified_content)

        #return f"파일이 성공적으로 수정되어 {output_file_path}에 저장되었습니다."
        return True, output_file_path

    except Exception as e:
        return f"예기치 못한 오류가 발생했습니다: {str(e)}"
    

def display_file_content(file):
    # 파일 내용을 읽어와 InputBox에 표시하는 함수
    if file is not None:
        try:
            with open(file.name, "r", encoding="utf-8") as f:  # 파일 경로를 열어서 읽음
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    return ""  # 파일이 없으면 빈 문자열 반환


def display_file_content(file):
    # 파일 내용을 읽어와 InputBox에 표시하는 함수
    if file is not None:
        try:
            with open(file.name, "r", encoding="utf-8") as f:  # 파일 경로를 열어서 읽음
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    return ""  # 파일이 없으면 빈 문자열 반환

def save_uploaded_file(uploaded_file, flag):
    """
    Gradio에서 업로드한 파일을 HOST_INPUT_DIR에 저장
    :param uploaded_file: gr.File 컴포넌트로 업로드된 파일 (단일 파일)
    :return: 저장 성공 시 저장된 파일 경로 메시지, 실패 시 에러 메시지.
    """
    if not uploaded_file:
        return "파일이 업로드되지 않았습니다."

    file_name = os.path.basename(uploaded_file.name)
    file_name = file_name[:-4].upper() + ".pdb"
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

def prodigy_process(file_obj):
    print("prodigy_process진입")
    #usr_dir = "static/prodigy"
    #dir_gen = "static/prodigy_gen"
    dir_target = "static/prodigy_target"
    
    """
    그라디오를 통해 업로드된 파일을 특정 경로에 저장하고 prodigy 스크립트를 실행하는 함수
    
    Args:
        file_obj: 그라디오 파일 업로드 컴포넌트에서 반환된 파일 객체
        text: 추가 텍스트 입력
    
    Returns:
        결과 데이터프레임 (행렬 형태로 변환됨 )
    """
    # 저장 디렉토리가 없으면 생성
    os.makedirs(dir_target, exist_ok=True)
    
    # 원본 파일 이름 추출
    original_filename = os.path.basename(file_obj.name)
    original_filename = original_filename.replace(" ", "")
    
    # 저장할 파일의 전체 경로
    save_path = os.path.join(dir_target, original_filename)
    
    # 파일 복사
    shutil.copy2(file_obj.name, save_path)
    print('save_path:', save_path)


    # 압축 파일 처리 (.zip)
    if save_path.lower().endswith('.zip'):
        # 압축 파일을 풀 디렉토리 생성
        extract_dir = os.path.join(dir_target, os.path.splitext(original_filename)[0])
        #os.makedirs(extract_dir, exist_ok=True)
        print("extract_dir:", extract_dir)

        try:
            # 압축 파일 유형에 따라 압축 해제
            # if save_path.lower().endswith('.zip'):
            #     import zipfile
            #     with zipfile.ZipFile(save_path, 'r') as zip_ref:
            #         zip_ref.extractall(extract_dir)
            if save_path.lower().endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(save_path, 'r') as zip_ref:
                    # 압축 파일 내 모든 항목 확인
                    for zip_info in zip_ref.infolist():
                        # 파일명만 추출 (경로 제거)
                        filename = os.path.basename(zip_info.filename)
                        # 파일명이 있고 디렉토리가 아닌 경우만 추출
                        if filename and not filename.endswith('/'):
                            # 원래 경로 정보 저장
                            zip_info.filename = filename
                            # 수정된 경로로 파일 추출
                            zip_ref.extract(zip_info, extract_dir)
            
            # 추출된 디렉토리에서 .pdb 파일 찾기
            pdb_files = []
            for root, _, files in os.walk(extract_dir):
                for file in files:
                    if file.lower().endswith('.pdb') and not file.lower().endswith('_prefix.pdb'):
                        pdb_files.append(os.path.join(root, file))
                        
            print("pdb_files:", pdb_files)
            # 각 PDB 파일에 대해 처리 결과 저장
            all_results = []
            num_size = 0
            for i, pdb_file in enumerate(pdb_files):
                print("ㅡㅡㅡㅡ")
                pdb_save_path = pdb_file
                pdb_output_path = pdb_file + "_prefix.pdb"
                
                # 체인 이름 변경
                rename_chains_by_header(pdb_save_path, pdb_output_path)
                print("pdb_output_path:", pdb_output_path)
                
                original_pdb_output_path = os.path.basename(pdb_output_path)
                original_pdb_output_path = original_pdb_output_path.replace(" ", "")
                print("original_pdb_output_path:", original_pdb_output_path)

                # prodigy 실행
                metrics, values = run_prodigy(pdb_output_path, original_pdb_output_path)
                print("metrics:", metrics)
                print("values:", values)
                
                num_size += 1
                all_results.append(values)

            print("all_results::", all_results)
            print("num_size::", num_size)

            # 압축 파일과 압축 해제된 디렉토리 내 .pdb 파일만 삭제
            try:
                # 원본 zip 파일 삭제
                os.remove(save_path)
                print(f"Deleted zip file: {save_path}")
                
                # 압축 해제된 디렉토리에서 .pdb 파일만 삭제
                for root, _, files in os.walk(extract_dir):
                    for file in files:
                        if file.lower().endswith('.pdb') and not file.lower().endswith('_prefix.pdb'):
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                            print(f"Deleted PDB file: {file_path}")
                
                print(f"Deleted all PDB files in directory: {extract_dir}")
            except Exception as e:
                print(f"Error while deleting files: {e}")

            output_html = render_pdbs_by_prefix_prodigy(extract_dir, num_size, 0)
            
            return all_results, output_html
        
        except Exception as e:
            print(f"Error extracting or processing archive: {e}")
            # 오류 발생 시에도 압축 파일과 압축 해제된 디렉토리 삭제
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                if os.path.exists(extract_dir):
                    shutil.rmtree(extract_dir)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {cleanup_error}")

            return [["Error"], [f"Archive extraction failed: {str(e)}"]], ""
        
    else:    
        # 일반 PDB 파일 처리
        rename_chains_by_header(save_path, save_path + "_prefix.pdb")
        os.remove(save_path)
        save_path = save_path + "_prefix.pdb"
        
        return run_prodigy(save_path, original_filename), render_pdbs_by_prefix_prodigy(save_path, 1)

    
def run_prodigy(pdb_path, original_filename):
    """
    PRODIGY 스크립트를 실행하고 결과를 파싱하는 함수
    
    Args:
        pdb_path: PDB 파일 경로
    
    Returns:
        결과 데이터프레임 (행렬 형태로 변환됨)
    """
    # 주어진 id로 외부 스크립트 실행
    script_command = f"prodigy {pdb_path} --selection A B"
    
    try:
        result = subprocess.run(script_command, shell=True, check=True, text=True, capture_output=True)
        print("result:", result.stdout)
        
        # 결과 파싱
        lines = result.stdout.splitlines()
        
        # 첫 번째 [+] 라인은 건너뛰기
        start_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("[+]"):
                start_idx = i + 1
                break
        
        # 데이터 추출
        metrics = ['[File Name]']
        original_filename = original_filename.replace(".pdb_prefix", "")
        values = [f"{original_filename}"]
        
        for line in lines[start_idx:]:
            if line.startswith("[+]") or line.startswith("[++]"):
                # 콜론(:)으로 분리
                parts = line.split(":", 1)
                
                if len(parts) == 2:
                    # [+] 또는 [++] 제거하고 앞뒤 공백 제거
                    metric_name = parts[0].replace("[+]", "").replace("[++]", "").strip()
                    
                    # 값 부분에서 숫자 추출
                    value_part = parts[1].strip()
                    
                    # 숫자 값 추출 (과학적 표기법 포함)
                    try:
                        # 첫 번째 숫자만 추출
                        value_match = re.search(r'[-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?', value_part)
                        if value_match:
                            value = float(value_match.group(0))
                            metrics.append(metric_name)
                            values.append(value)
                    except:
                        # 숫자 추출 실패 시 원본 값 사용
                        metrics.append(metric_name)
                        values.append(value_part)
        
        # 인덱스 10과 11의 요소를 인덱스 1과 2로 이동
        if len(values) > 11:
            # 인덱스 10의 요소 추출 후 제거
            value_10 = values.pop(10)
            # 인덱스 11의 요소가 인덱스 10으로 이동했으므로 다시 10에서 추출
            value_11 = values.pop(10)
            
            # 인덱스 1과 2에 삽입 (역순으로 삽입해야 원하는 순서가 됨)
            values.insert(1, value_10)
            values.insert(2, value_11)

        if len(metrics) > 11:
            # 인덱스 10의 요소 추출 후 제거
            metric_10 = metrics.pop(10)
            # 인덱스 11의 요소가 인덱스 10으로 이동했으므로 다시 10에서 추출
            metric_11 = metrics.pop(10)
            
            # 인덱스 1과 2에 삽입 (역순으로 삽입해야 원하는 순서가 됨)
            metrics.insert(1, metric_10)
            metrics.insert(2, metric_11)
        
        metrics[1] = "Predicted binding affinity (ΔG) (kcal.mol-1)"
        metrics[2] = "Predicted dissociation constant (Kd) (M) at 25.0˚C"
        
        # 데이터프레임 형식으로 변환 (행렬 형태)
        # 첫 번째 행은 지표 이름, 두 번째 행은 값
        print("[metrics, values]::", [metrics, values])
        return [metrics, values]
        
    except subprocess.CalledProcessError as e:
        print(f"Error running script: {e}")
        return [["Error"], [str(e)]]
    except Exception as e:
        print(f"Unexpected error: {e}")
        return [["Error"], [str(e)]]
    

def extract_unique_values_from_column(file_path, column_index=4):
    unique_values = set()
    
    with open(file_path, 'r') as file:
        for line in file:
            # 공백으로 분리하여 열 데이터 추출
            columns = line.split()
            
            # 열의 개수가 충분한지 확인
            if len(columns) > column_index:
                unique_values.add(columns[column_index])
    
    return unique_values

def rename_chains_by_header(input_pdb, output_pdb):
    """
    ClusPro 출력 PDB 파일에서 "HEADER lig" 문자열을 기준으로 두 chain의 ID를 변경하는 함수.
    Chain A는 'A', Chain B는 'B'로 변경.
    결과는 단일 PDB 파일로 출력.

    Args:
        input_pdb: 입력 PDB 파일 경로.
        output_pdb: 출력 PDB 파일 경로.
    """
    try:
        with open(input_pdb, 'r') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_pdb}'를 찾을 수 없습니다.")
        return

    split_index = -1
    for i, line in enumerate(lines):
        if line.startswith("HEADER lig"):
            split_index = i
            break

    if split_index == -1:
        print("오류: 'HEADER lig' 문자열을 찾을 수 없습니다. PDB 파일 형식을 확인하세요.")
        return

    # Chain ID 변경
    for i, line in enumerate(lines):
        if line.startswith("ATOM") or line.startswith("HETATM"):
            if i < split_index:  # HEADER lig 이전 (Chain A)
                lines[i] = line[:21] + 'A' + line[22:]
            else:  # HEADER lig 이후 (Chain B)
                lines[i] = line[:21] + 'B' + line[22:]

    # TER 레코드 Chain ID도 변경
    for i, line in enumerate(lines):
      if line.startswith("TER"):
        if i < split_index:
          lines[i] = line[:21] + 'A' + line[22:]
        else:
          lines[i] = line[:21] + 'B' + line[22:]

    # 수정된 PDB 파일 저장
    with open(output_pdb, 'w') as outfile:
        outfile.writelines(lines)






###############################
# DeepBSRPred Tab 관련 함수
def run_pred(file, flag):
    """
    DeepBSRPred run
    """
    pdb_code = os.path.basename(file)
    pdb_path = os.path.join(HOST_INPUT_DIR, pdb_code)
    print("pdb_path:", pdb_path)
    pdb_code = pdb_code.split(".")[0]
    print("pdb_code:", pdb_code)
    
    if flag==1: #파일업로드 일경우에만 수행
        print('flag==1')
        save_uploaded_file2(file, flag)
    if flag==2:
        print('flag==2')
        pdb_path = pdb_path + ".pdb"
    
    print('send_to_flask_server으로 보낼때:pdb_code:',pdb_code)
    print('send_to_flask_server으로 보낼때:pdb_path:',pdb_path)
    scannet_output = send_to_flask_server(pdb_code, pdb_path) + ' '
    print("scannet_output:", scannet_output)

    file_pred = f"./file/{pdb_code}/predictions_{pdb_code}.csv"
    tmp_df = pd.read_csv(file_pred)
    chains = tmp_df["Chain"].unique()
    
    venv_path = "~/miniforge3/bin/activate"
    venv_path = "/home/eps/venv/Gradio/bin/activate"
    out_log = []
    input_file_path = '/mnt/c/Users/user/Desktop/FinalProject/Gradio/input/'
    input_file_path = '/home/eps/prj_envs/Gradio/input/'
    DeepBSR_path = '/home/eps/prj_envs/Gradio/DeepBSRPred/'
    for c in chains:
        command = f"source {venv_path} && conda activate Gradio && cd /mnt/d/Final/DeepBSRPred/ && python3 feature_calculation_prediction_ver1.py --input_file {input_file_path}{pdb_code}.pdb "
        command = f"source {venv_path} && cd {DeepBSR_path} && python {DeepBSR_path}feature_calculation_prediction_ver1.py --input_file {input_file_path}{pdb_code}.pdb "
        print("function.py임임")
        print("command:",command)
        # subprocess 실행
        result = subprocess.run(command, shell=True, capture_output=True, text=True, executable="/bin/bash")

        # 실행 결과 출력
        # print("STDOUT:\n", result.stdout)
        # print("STDERR:\n", result.stderr)
        out_log.append(f"{result.stderr if result.stderr else 'DeepBSRPred 모델 수행완료'}")
    
    return scannet_output + 'DeepBSRPred 모델 수행완료'

# Weight 기반 combined_avg 열 생성
def run_merge_csv(pdb_code, scannet_weight, bsrp_weight):
    """
    ScanNet and DeepBSRPred Result Combine
    """ 

    # 파일 경로
    file_pred = f"./file/{pdb_code}/predictions_{pdb_code}.csv"
    file_bsrp = f"./file/{pdb_code}/{pdb_code}_DeepBSRPred_result.csv"

    try:
        # 데이터 로드
        df_pred = pd.read_csv(file_pred)
        df_bsrp = pd.read_csv(file_bsrp)
        
        # Check if the Residue column from df1 matches the Sequence column from df2
        if list(df_bsrp['Residue']) == list(df_pred['Sequence']):
            # If they match, add the Prediction column to df2
            df_pred['Prediction'] = df_bsrp['Prediction']
            
            # Save the merged file
            # print(df2.head(10))
            # df_pred.to_csv('merged_file.csv', index=False)
            print("Files merged successfully. The residue sequences match.")
        else:
            print("Error: The residue sequences do not match.")
            # Optional: Find the first mismatch
            for i, (res1, res2) in enumerate(zip(df_bsrp['Residue'], df_pred['Sequence'])):
                if res1 != res2:
                    print(f"First mismatch at position {i+1}: {res1} vs {res2}")
                    break

        # 3단계: 최종 데이터 정렬 (Residue Index 기준 오름차순 정렬)
        df_result = df_pred[["Chain", "Residue Index", "Sequence", "Binding site probability", "Prediction"]]
        df_result["combine_pred"] = weight_avg(df_result["Binding site probability"], df_result["Prediction"], scannet_weight, bsrp_weight)
        df_result = df_result.sort_values(by=["Chain", "Residue Index"]).reset_index(drop=True)

        # 결과 저장
        df_result.to_csv(f"./file/{pdb_code}/{pdb_code}_filtered_results.csv", index=False)
        print(f"처리 완료! 결과가 ./file/{pdb_code}/{pdb_code}_filtered_({scannet_weight}_{bsrp_weight}).csv에 저장되었습니다.")
        print(type(df_result))
        
        return df_result
    
    except Exception as e:
        print("Exception")
        return pd.DataFrame({"Error": [str(e)]})

# Weight 계산 (가중산술평균)
def weight_avg(pred_scan, pred_prob, scannet_weight, bsr_weight):
    w_scan, w_prob = scannet_weight, bsr_weight
    return (w_scan * pred_scan + w_prob * pred_prob) / (w_scan + w_prob)
    
# Protein Binding Site Analysis Func
def filter_residues_by_weight(df, k, threshold):
    print(df.head())
    df = df.sort_values(by=["Chain", "Residue Index"]).reset_index(drop=True)
    valid_groups = []

    for i in range(len(df) - k + 1):
        subset = df.iloc[i:i + k]
        
        if len(subset["Chain"].unique()) > 1:
            continue

        if not all(subset["Residue Index"].iloc[j] == subset["Residue Index"].iloc[j - 1] + 1 for j in range(1, k)):
            continue

        avg_weight = subset["combine_pred"].mean()
        if avg_weight >= threshold:
            valid_groups.append((subset.reset_index(drop=True), avg_weight))
    print(valid_groups)
    return valid_groups

def update_weights(scannet_value):
    return round(1 - scannet_value, 3)  # deepBSR_weight = 1 - scannet_weight

# Protein Binding Site Analysis Func
def process_uploaded_file(file, group_size, threshold_input, scannet_weight, bsrp_weight):
    ('진입1')
    pdb_code = os.path.basename(file)
    pdb_code = pdb_code.split(".")[0]
    print(pdb_code)
    molstar_output = generate_molstar_html_scannet(pdb_code)
    df = run_merge_csv(pdb_code, scannet_weight, bsrp_weight)
    # df = pd.read_csv(file.name)
    
    print(df.head())
    result_groups = filter_residues_by_weight(df, group_size, threshold_input)
    summary_data = []
    global detailed_data_dict
    detailed_data_dict = {}
    
    # Create unsorted summary data
    for idx, (group, weight_avg) in enumerate(result_groups):
        contig = f"{group['Chain'].iloc[0]}{group['Residue Index'].iloc[0]}-{group['Residue Index'].iloc[-1]}"
        summary_data.append({"Contig": contig, "AVG": round(weight_avg, 4)})
        detailed_data_dict[contig] = group
    
    # Sort by AVG in descending order
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="AVG", ascending=False).reset_index(drop=True)
    summary_df.to_csv(f'./file/{pdb_code}/Summary_{group_size}_{threshold_input}.csv', index=False)
    # Add Rank column after sorting
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))
    
    df_new_column_names = ['사슬', '잔기 번호', '시퀀스', '결합 부위 확률', '예측 확률', '예측 결합']
    df.columns = df_new_column_names

    summary_df_new_column_names = ["순위(Rank)", "컨티그(Contig)", "평균값(AVG)"]
    summary_df.columns = summary_df_new_column_names

    return (
        summary_df,
        gr.update(value=f'./file/{pdb_code}/Summary_{group_size}_{threshold_input}.csv', visible=True),
        gr.Dataframe(value=df, visible=True),
        molstar_output
    )

def process_uploaded_file2(inputbox_code, group_size, threshold_input, scannet_weight, bsrp_weight):
    ('진입2')
    pdb_code = inputbox_code.upper()
    print(pdb_code)
    molstar_output = generate_molstar_html_scannet(pdb_code)
    df = run_merge_csv(pdb_code, scannet_weight, bsrp_weight)
    # df = pd.read_csv(file.name)
    
    print(df.head())
    result_groups = filter_residues_by_weight(df, group_size, threshold_input)
    summary_data = []
    global detailed_data_dict
    detailed_data_dict = {}
    
    print(result_groups)
    # Create unsorted summary data
    for idx, (group, weight_avg) in enumerate(result_groups):
        contig = f"{group['Chain'].iloc[0]}{group['Residue Index'].iloc[0]}-{group['Residue Index'].iloc[-1]}"
        summary_data.append({"Contig": contig, "AVG": round(weight_avg, 4)})
        detailed_data_dict[contig] = group
    
    # Sort by AVG in descending order
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values(by="AVG", ascending=False).reset_index(drop=True)
    summary_df.to_csv(f'./file/{pdb_code}/Summary_{group_size}_{threshold_input}.csv', index=False)
    # Add Rank column after sorting
    summary_df.insert(0, "Rank", range(1, len(summary_df) + 1))

    df_new_column_names = ['사슬', '잔기 번호', '시퀀스', '결합 부위 확률', '예측 확률', '예측 결합']
    df.columns = df_new_column_names

    summary_df_new_column_names = ["순위(Rank)", "컨티그(Contig)", "평균값(AVG)"]
    summary_df.columns = summary_df_new_column_names
    
    return (
        summary_df,
        gr.update(value=f'./file/{pdb_code}/Summary_{group_size}_{threshold_input}.csv', visible=True),
        gr.Dataframe(value=df, visible=True),
        molstar_output
    )

# Protein Binding Site Analysis Func
def show_details(evt: gr.SelectData, df):
    """선택한 Contig에 해당하는 데이터만 반환"""
    selected_index = evt.index[0] if isinstance(evt.index, list) else evt.index
    #contig = df.iloc[selected_index]["Contig"]
    contig = df.iloc[selected_index]["컨티그(Contig)"]
    print(f"Selected contig: {contig}")

    result_detail_df=detailed_data_dict.get(contig, pd.DataFrame())
    result_detail_df=result_detail_df.rename(columns={
        'Chain': '사슬',
        'Residue Index': '잔기 번호',
        'Sequence': '시퀀스',
        'Binding site probability': '결합 부위 확률',
        'pred_prob': '예측 확률',
        'combine_pred': '예측 결합'
    })

    return result_detail_df
    
# 데이터프레임 업데이트
def update_data_table(input_id):
    # input_id를 기반으로 CSV 파일 처리 로직 추가
    input_id = input_id.upper()
    #print('input_id:', input_id)

    try:
        # 예시: 특정 파일 경로와 열 이름을 사용하여 데이터프레임 생성
        file_path = f"./file/{input_id}/predictions_{input_id}.csv"  # input_id를 사용하여 파일 경로 생성
        print(file_path)
        sort_column = "Binding site probability"  # 정렬 기준 열 이름 (예시)
        sorted_df = load_and_sort_csv(file_path, sort_column)
        return sorted_df
    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})  # 에러 발생 시 에러 메시지를 데이터프레임으로 반환
    


def click_pred_scannet(pdb_code, pdb_file, flag):
    """입력된 값이 PDB 코드인지 파일인지 판별 후 적절한 함수 실행"""
    if pdb_code is None or pdb_code.strip() == "":
        flag = 1
        return run_pred(pdb_file, flag)
    else:
        print('get_pdb_file2진입')
        pdb_code_upper = get_pdb_file2(pdb_code, flag)
        flag = 2
        print('pdb_code_upper:::',pdb_code_upper)
        return run_pred(pdb_code_upper, flag)

def click_process_uploaded_file(input_box_3scannet, file_input, group_size, threshold_input, scannet_weight, bsrp_weight):
    """입력된 값이 PDB 코드인지 파일인지 판별 후 적절한 함수 실행"""
    if input_box_3scannet is None or input_box_3scannet.strip() == "":
        return process_uploaded_file(file_input, group_size, threshold_input, scannet_weight, bsrp_weight)
    else:
        return process_uploaded_file2(input_box_3scannet, group_size, threshold_input, scannet_weight, bsrp_weight)


def save_uploaded_file2(uploaded_file, flag):
    """
    Gradio에서 업로드한 파일을 HOST_INPUT_DIR에 저장
    :param uploaded_file: gr.File 컴포넌트로 업로드된 파일 (단일 파일)
    :return: 저장 성공 시 저장된 파일 경로 메시지, 실패 시 에러 메시지.
    """
    if not uploaded_file:
        return "파일이 업로드되지 않았습니다."

    file_name = os.path.basename(uploaded_file.name)
    #file_name = file_name[:-4].upper() + ".pdb"
    file_name = file_name[:-4] + ".pdb"
    # 파일 저장 경로
    #if flag:
        #target_path = os.path.join(HOST_RFDIFFUSION_OUTPUT_DIR, file_name)
    #else:
    target_path = os.path.join(HOST_INPUT_DIR, file_name)

    try:
        shutil.copy(uploaded_file.name, target_path)
        os.chmod(target_path, 0o755)
        return f"✅ 업로드 완료: {target_path}", file_name
    except Exception as e:
        return f"❌ 업로드 중 에러 발생: {e}"