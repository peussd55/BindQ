from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import os
import zipfile
import logging
import requests
import uvicorn
from typing import Dict
import shutil

app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.DEBUG)

# ZIP 파일 저장 디렉토리
FILES_DIR = "./files"
os.makedirs(FILES_DIR, exist_ok=True)  # 디렉토리가 없으면 생성

class PredictRequest(BaseModel):
    id: str
    pdb_path: str

def download_file(url: str, save_dir: str, file_name: str) -> str:
    """주어진 URL에서 파일을 다운로드하고 지정된 경로에 저장"""
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    file_path = os.path.join(save_dir, file_name)
    print(file_path)
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        return file_path
    else:
        raise Exception(f"파일 다운로드 실패: {response.status_code}")
    
def get_file(pdb_path: str, save_dir: str, file_name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)  # 디렉토리가 없으면 생성
    file_path = os.path.join(save_dir, file_name)
    print(file_path)
    try:
        shutil.copy(pdb_path, file_path)
        return file_path
    except Exception as e:
        return f"복사실패: {e}"

@app.post("/predict")
async def predict(request: PredictRequest) -> FileResponse:
    id = request.id
    logging.info(f"Received ID: {id}")
    pdb_path = request.pdb_path
    logging.info(f"Received ID: {pdb_path}")
    flag = 1

    try:
        pdb_url = f"https://files.rcsb.org/download/{id}.pdb"
        if len(id) == 4:
            pdb_url = f"https://files.rcsb.org/download/{id}.pdb"
        elif len(id) == 6:
            pdb_url = f"https://alphafold.ebi.ac.uk/files/AF-{id}-F1-model_v4.pdb"

        # RCSB PDB에서 파일 다운로드
        #id = id.upper()
        save_dir = "./files"
        #save_dir = "/home/eps/prj_envs/ScanNet/files"
        file_name = f"{id}.pdb"

        if flag==0:
            logging.info(f"Downloading PDB file from: {pdb_url}")
            downloaded_file_path = download_file(pdb_url, save_dir, file_name)
            logging.info(f"PDB 파일이 다운로드되었습니다: {downloaded_file_path}")
        else:
            downloaded_file_path = get_file(pdb_path, save_dir, file_name)
            logging.info(f"PDB 파일이 복사되었습니다: {downloaded_file_path}")

        # 주어진 id로 외부 스크립트 실행
        script_command = f"python predict_bindingsites.py {downloaded_file_path} --noMSA"
        logging.info(f"Running command: {script_command}")
        subprocess.run(script_command, shell=True, check=True)

        # 생성된 파일 경로 정의
        predictions_dir = os.path.join("predictions", f"{id}_single_ScanNet_interface_noMSA")
        logging.info(f"Checking predictions directory: {predictions_dir}")
        if not os.path.exists(predictions_dir):
            logging.error(f"Predictions folder not found: {predictions_dir}")
            raise HTTPException(status_code=500, detail=f"ID '{id}'에 대한 예측 폴더를 찾을 수 없습니다.")

        # 예상되는 파일 목록
        expected_files = [
            f"annotated_{id}.pdb",
            f"annotated_{id}.cxc",
            f"predictions_{id}.csv"
        ]

        # annotated_{id}.pdb가 없으면 annotated_{id}.cif로 대체
        pdb_path = os.path.join(predictions_dir, f"annotated_{id}.pdb")
        cif_path = os.path.join(predictions_dir, f"annotated_{id}.cif")
        if not os.path.exists(pdb_path):
            if os.path.exists(cif_path):
                logging.warning(f"PDB file missing, using CIF file instead: {cif_path}")
                expected_files[0] = f"annotated_{id}.cif"
            else:
                logging.error(f"Both PDB and CIF files are missing for ID: {id}")
                raise HTTPException(status_code=500, detail=f"annotated_{id}.pdb와 annotated_{id}.cif 파일이 모두 없습니다.")

        # 모든 파일이 존재하는지 확인
        missing_files = [file for file in expected_files if not os.path.exists(os.path.join(predictions_dir, file))]
        if missing_files:
            logging.error(f"Missing files: {missing_files}")
            raise HTTPException(status_code=500, detail=f"누락된 파일: {', '.join(missing_files)}")

        # ZIP 파일 생성 (디스크에 저장)
        zip_file_path = os.path.join(FILES_DIR, f"{id}_predictions.zip")
        with zipfile.ZipFile(zip_file_path, 'w') as zip_file:
            for file_name in expected_files:
                file_path = os.path.join(predictions_dir, file_name)
                logging.info(f"Adding file to ZIP: {file_path}")
                zip_file.write(file_path, arcname=file_name)

        # ZIP 파일을 응답으로 전송
        return FileResponse(
            zip_file_path,
            media_type='application/zip',
            filename=f"{id}_predictions.zip"
        )

    except subprocess.CalledProcessError as e:
        logging.error(f"Script execution failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"스크립트 실행 실패: {str(e)}")
    except Exception as e:
        logging.exception("An unexpected error occurred")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5002)
