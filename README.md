# BindQ
> 🧬 **단백질 결합부위 탐지부터 서열 설계, 도킹 검증까지 원스톱으로 지원하는 구조 기반 결합제 설계 파이프라인**

## 🛠️ 기술 스택 (Tech Stack)
<p>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white">
  <img src="https://img.shields.io/badge/Gradio-FF7C00?style=for-the-badge&logo=gradio&logoColor=white">
  <img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white">
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white">
  <img src="https://img.shields.io/badge/Mol*-00A9E0?style=for-the-badge&logoColor=white">
</p>

## 👥 참여인원 (Members)
<p>
  <img src="https://img.shields.io/badge/개발%20(Development)-2명-3776AB?style=flat-square&logo=github&logoColor=white">
  <img src="https://img.shields.io/badge/연구%20(Research)-3명-EE4C2C?style=flat-square&logo=googlescholar&logoColor=white">
</p>

> **📘 통합 파이프라인 요약** : [BindQ Wiki](https://github.com/peussd55/BindQ/wiki/BindQ-%ED%86%B5%ED%95%A9-%ED%8C%8C%EC%9D%B4%ED%94%84%EB%9D%BC%EC%9D%B8-%EC%9A%94%EC%95%BD)

## 프로젝트 개요
BindQ는 구조 기반 결합제 설계를 위한 여러 연구 도구를 하나의 FastAPI · Gradio 애플리케이션으로 통합한 플랫폼입니다. 사이드바 기반의 다섯 가지 탭(Home, 결합부위 예측, RFdiffusion, ProteinMPNN, Prodigy)에서 전체 설계 프로세스를 순차적으로 실행할 수 있으며, 실시간 채팅형 헬퍼와 정적 자산 제공 기능을 함께 제공합니다.

## 주요 기능
### 1. 구조기반 결합부위 예측
- PDB ID 또는 UniProt ID 입력 및 파일 업로드를 지원하며, ScanNet(원격 Flask API)과 DeepBSRPred를 동시에 호출해 결합 부위 확률을 산출합니다.
- 두 모델의 결과를 가중 평균하여 `combine_pred` 열을 생성하고, 연속된 잔기 그룹에 대한 가중 평균을 기반으로 상위 결합 후보를 정렬합니다.
- 사용자는 결합부위 리스트를 선택해 상세 표를 확인하고, Mol* 기반 3D 뷰어에서 구조를 시각화할 수 있습니다.

### 2. De novo 단백질 백본 설계 (RFdiffusion)
- 업로드한 백본 또는 외부에서 다운로드한 PDB를 기반으로 컨티그, 핫스팟, 생성 횟수 등의 파라미터를 설정합니다.
- 입력 파라미터는 JSON 플래그로 변환된 뒤 `rfdiffusion:latest` Docker 컨테이너를 `--gpus all` 옵션으로 실행하여 결과 구조를 생성합니다.
- 생성된 PDB는 컨티그/핫스팟 별 하위 디렉터리로 이동하며, 3Dmol.js 뷰어와 다운로드 링크가 동시에 제공됩니다.

### 3. 후보물질 서열 생성 (ProteinMPNN)
- RFdiffusion 출력 구조를 입력으로 사용하여 ProteinMPNN Docker 컨테이너를 호출하고, 다중 서열 샘플과 확률 텐서를 계산합니다.
- RMSD, pLDDT 비교 그래프, 확률 히트맵, CSV 다운로드 등 후처리 산출물을 자동으로 생성합니다.
- Gradio 탭에는 ProteinMPNN 전용 UI(`get_mpnn_ui`)가 그대로 노출되어 고급 파라미터를 조정할 수 있습니다.

### 4. 도킹 및 결합 친화도 검증 (PRODIGY)
- 업로드한 단일 PDB 또는 ZIP 묶음에 대해 체인 명시를 자동 보정한 뒤 PRODIGY CLI를 실행하여 ΔG, Kd 및 다양한 상호작용 지표를 추출합니다.
- 분석 결과는 표 형태로 반환되며, 정리된 PDB를 3Dmol.js 뷰어에서 확인할 수 있습니다.
- ClusPro 링크 버튼과 안내 이미지를 함께 제공하여 외부 도킹 도구 연계를 돕습니다.

### 5. 보조 기능
- 사이드바에 위치한 BindQ Chatbot으로 간단한 질의응답을 처리하고, 정적 자산(`static/`)을 FastAPI 정적 경로로 제공해 Mol*, 3Dmol.js 등의 클라이언트 리소스를 서빙합니다.

## 시스템 구성 및 데이터 흐름
1. **데이터 수집** – PDB/UniProt ID 또는 사용자 업로드를 처리하여 입력 구조를 준비합니다.
2. **결합부위 분석** – ScanNet Flask API에서 ZIP 결과를 수신하고(`send_to_flask_server`), DeepBSRPred 스크립트를 로컬 실행한 뒤, 가중 평균과 연속 잔기 필터링을 통해 후보 영역을 도출합니다.
3. **백본 생성** – RFdiffusion Docker 컨테이너에 override 플래그를 전달하여 다중 설계 구조를 생성하고, 결과를 컨티그/핫스팟 기준으로 정리합니다.
4. **서열 설계** – ProteinMPNN 컨테이너로 서열을 샘플링하고, RMSD·pLDDT·확률 히트맵을 포함한 분석 결과를 생성합니다.
5. **결합도 검증** – PRODIGY CLI 실행을 통해 도킹 구조의 결합 친화도를 계산하고, 3D 시각화와 요약 표를 제공합니다.

## 디렉토리 구성
- `Gradio/`: FastAPI·Gradio 앱, 파이프라인 제어 함수, 실행 스크립트 및 정적 자산이 위치합니다.
- `DeepBSRPred/`: DeepBSRPred 원본 스크립트와 의존 자원이 포함되어 있으며, Gradio 함수에서 해당 스크립트를 호출합니다.
- `ProteinMPNN/`: ProteinMPNN 앱과 Dockerfile이 포함되어 있으며, 컨테이너 내부에서 `app.py` UI를 재사용합니다.
- `RFdiffusion/`: RFdiffusion 원본 코드와 Dockerfile, conda 환경 정의가 포함되어 있습니다.
- `ScanNet/`: ScanNet 관련 학습·추론 스크립트가 포함되어 있으며, 외부 Flask 서비스 구현 시 활용됩니다.
- `dssp/`, `freesasa/`: DeepBSRPred가 필요로 하는 외부 도구 소스가 포함되어 있으며, DeepBSRPred 스크립트에서 해당 라이브러리를 직접 사용합니다.

## 환경 준비
1. **Gradio 애플리케이션 의존성 설치**
   ```bash
   pip install -r Gradio/requirements.txt
   ```
   위 요구 사항 파일에는 FastAPI, Gradio, BioPython, Ray, Plotly, TensorFlow, PyTorch 등 전체 파이프라인 실행에 필요한 패키지가 정의되어 있습니다.

2. **Docker 이미지 준비**
   - RFdiffusion: `RFdiffusion/Dockerfile`을 사용하여 CUDA 11.7 기반 이미지와 conda 환경을 구성합니다.
   - ProteinMPNN: `ProteinMPNN/Dockerfile`로 Python 환경과 애플리케이션을 설정합니다.

3. **호스트 경로 및 서비스 엔드포인트 수정**
   - `Gradio/run_pipeline.py` 상단의 `HOST_*` 상수는 실제 배포 경로에 맞게 조정해야 합니다.
   - ScanNet Flask API 주소(`FLASK_SERVER_URL`)를 환경에 맞게 수정하십시오.

4. **외부 도구 설치**
   DeepBSRPred 실행에는 DSSP, MSMS, HBPLUS, FreeSASA 등이 필요하므로 `/dssp`, `/freesasa` 등의 소스를 참고하여 사전 설치합니다.

## 실행 방법
1. 필수 Docker 컨테이너(RFdiffusion, ProteinMPNN)를 가동 가능한 상태로 준비합니다.
2. ScanNet 예측을 위한 Flask 백엔드를 실행하거나, 제공된 엔드포인트를 이용합니다.
3. FastAPI 서버를 구동합니다.
   ```bash
   cd Gradio
   uvicorn app:app --host 0.0.0.0 --port 5001 --reload
   ```
   애플리케이션은 `/gradio` 경로로 Gradio UI를 마운트하며, `static/`과 `rfdiffusion_output/` 디렉터리가 FastAPI 정적 경로로 노출됩니다.
4. 브라우저에서 `http://<host>:5001/gradio`로 접속한 뒤, 탭을 순차적으로 실행하여 결합부위 탐색 → 백본 설계 → 서열 설계 → 도킹 검증을 진행합니다.

## 운영 팁
- RFdiffusion 출력은 컨티그/핫스팟 별 폴더로 정리되므로, 동일한 입력에 대한 반복 실험 시 해당 디렉터리를 삭제하거나 별도 보관하세요.
- ProteinMPNN 결과(`proteinmpnn_output/probs/*.npz`)는 후속 분석을 위해 CSV로 자동 변환되므로, 외부 분석 파이프라인과 쉽게 연계할 수 있습니다.
- PRODIGY 분석 후에는 업로드한 ZIP 및 중간 PDB 파일이 자동 정리되지만, 실패 시 수동 삭제가 필요할 수 있습니다.

<img width="1338" height="633" alt="화면 캡처 2025-10-18 192549" src="https://github.com/user-attachments/assets/d9188200-ec90-4062-928f-f9c72ae9bdeb" />
<img width="1330" height="628" alt="화면 캡처 2025-10-18 192608" src="https://github.com/user-attachments/assets/644221e8-0c7b-41b1-947d-432842dc4adf" />
<img width="1336" height="629" alt="화면 캡처 2025-10-18 192620" src="https://github.com/user-attachments/assets/3b67ef6c-6b41-44ea-9fef-15f7477d22dc" />
<img width="1335" height="626" alt="화면 캡처 2025-10-18 192630" src="https://github.com/user-attachments/assets/8821ccce-b50b-478e-960a-b23f5019ac3b" />
<img width="1334" height="621" alt="화면 캡처 2025-10-18 192652" src="https://github.com/user-attachments/assets/0c26faa6-e9e4-4729-ab90-364e14d20e8a" />

## 의존성파일 다운방법
### 1. 모든 디렉토리를 내려받기
- ProteinMPNN 하위에 params 디렉토리를 생성하고 wget으로 파라미터를 다운로드받는다.

  `wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/LICENSE`
  
  `wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_1.npz`
  
  `wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_1_ptm.npz`
  
  `wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_5.npz`
  
  `wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_5_ptm.npz`

- RFdiffusion 하위에 models 디렉토리를 생성하고 wget으로 모델을 다운로드 받는다.

   `wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt`
 
   `wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt`

### 2. DeepBSRPred, ProteinMPNN, dssp, freesasa 4개의 디렉토리를 Gradio 디렉토리 하위로 이동시킨다.(같은 환경설정. Gradio/requirements.txt 참조하여 의존성 패키지 설치)
### 3. DeepBSRPred 의존성 패키지 별도 설치 : 
- dssp 설치
https://github.com/PDB-REDO/dssp

- msms 설치
https://ccsb.scripps.edu/msms/downloads/

- msms 설치 가이드
https://ssbio.readthedocs.io/en/latest/instructions/msms.html

- hbplus 설치
https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/install.html
### 4. RFdiffusion은 하위 Dockerfile로 이미지생성한다.
### 5. ScanNet은 별도의 conda 가상환경으로 기동한다. (ScanNet/requirements.txt, ScanNet/environmemt.yml 참조하여 의존성 패키지 설치)

### ***라이브러리 출처 :
  https://github.com/RosettaCommons/RFdiffusion/tree/main
  
  https://github.com/jertubiana/ScanNet
  
  https://huggingface.co/spaces/simonduerr/ProteinMPNN

  https://web.iitm.ac.in/bioinfo2/deepbsrpred/index.html

## 참고
- RFdiffusion, ProteinMPNN, ScanNet, DeepBSRPred는 각각의 원저자 저장소 라이선스를 따르므로 사용 전에 해당 프로젝트의 정책을 확인하십시오.
