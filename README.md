# BindQ
> 🧬 **단백질 결합부위 탐지부터 서열 설계, 도킹 검증까지 원스톱으로 지원하는 구조 기반 결합제 설계 파이프라인**

## 프로젝트 개요
BindQ는 구조 기반 결합제 설계를 위한 여러 연구 도구를 하나의 FastAPI · Gradio 애플리케이션으로 통합한 플랫폼입니다. 사이드바 기반의 다섯 가지 탭(Home, 결합부위 예측, RFdiffusion, ProteinMPNN, Prodigy)에서 전체 설계 프로세스를 순차적으로 실행할 수 있으며, 실시간 채팅형 헬퍼와 정적 자산 제공 기능을 함께 제공합니다.【F:Gradio/function.py†L19-L37】【F:Gradio/app.py†L200-L497】

## 주요 기능
### 1. 구조기반 결합부위 예측
- PDB ID 또는 UniProt ID 입력 및 파일 업로드를 지원하며, ScanNet(원격 Flask API)과 DeepBSRPred를 동시에 호출해 결합 부위 확률을 산출합니다.【F:Gradio/app.py†L270-L338】【F:Gradio/function.py†L490-L536】【F:Gradio/function.py†L991-L1036】
- 두 모델의 결과를 가중 평균하여 `combine_pred` 열을 생성하고, 연속된 잔기 그룹에 대한 가중 평균을 기반으로 상위 결합 후보를 정렬합니다.【F:Gradio/function.py†L1039-L1155】
- 사용자는 결합부위 리스트를 선택해 상세 표를 확인하고, Mol* 기반 3D 뷰어에서 구조를 시각화할 수 있습니다.【F:Gradio/app.py†L311-L338】【F:Gradio/function.py†L1121-L1155】

### 2. De novo 단백질 백본 설계 (RFdiffusion)
- 업로드한 백본 또는 외부에서 다운로드한 PDB를 기반으로 컨티그, 핫스팟, 생성 횟수 등의 파라미터를 설정합니다.【F:Gradio/app.py†L360-L432】
- 입력 파라미터는 JSON 플래그로 변환된 뒤 `rfdiffusion:latest` Docker 컨테이너를 `--gpus all` 옵션으로 실행하여 결과 구조를 생성합니다.【F:Gradio/run_pipeline.py†L67-L295】
- 생성된 PDB는 컨티그/핫스팟 별 하위 디렉터리로 이동하며, 3Dmol.js 뷰어와 다운로드 링크가 동시에 제공됩니다.【F:Gradio/function.py†L117-L209】

### 3. 후보물질 서열 생성 (ProteinMPNN)
- RFdiffusion 출력 구조를 입력으로 사용하여 ProteinMPNN Docker 컨테이너를 호출하고, 다중 서열 샘플과 확률 텐서를 계산합니다.【F:Gradio/run_pipeline.py†L322-L431】
- RMSD, pLDDT 비교 그래프, 확률 히트맵, CSV 다운로드 등 후처리 산출물을 자동으로 생성합니다.【F:Gradio/run_pipeline.py†L372-L430】
- Gradio 탭에는 ProteinMPNN 전용 UI(`get_mpnn_ui`)가 그대로 노출되어 고급 파라미터를 조정할 수 있습니다.【F:Gradio/app.py†L437-L441】

### 4. 도킹 및 결합 친화도 검증 (PRODIGY)
- 업로드한 단일 PDB 또는 ZIP 묶음에 대해 체인 명시를 자동 보정한 뒤 PRODIGY CLI를 실행하여 ΔG, Kd 및 다양한 상호작용 지표를 추출합니다.【F:Gradio/function.py†L692-L913】
- 분석 결과는 표 형태로 반환되며, 정리된 PDB를 3Dmol.js 뷰어에서 확인할 수 있습니다.【F:Gradio/function.py†L293-L338】【F:Gradio/function.py†L692-L826】
- ClusPro 링크 버튼과 안내 이미지를 함께 제공하여 외부 도킹 도구 연계를 돕습니다.【F:Gradio/app.py†L444-L485】

### 5. 보조 기능
- 사이드바에 위치한 BindQ Chatbot으로 간단한 질의응답을 처리하고, 정적 자산(`static/`)을 FastAPI 정적 경로로 제공해 Mol*, 3Dmol.js 등의 클라이언트 리소스를 서빙합니다.【F:Gradio/app.py†L200-L247】【F:Gradio/app.py†L22-L24】

## 시스템 구성 및 데이터 흐름
1. **데이터 수집** – PDB/UniProt ID 또는 사용자 업로드를 처리하여 입력 구조를 준비합니다.【F:Gradio/function.py†L48-L90】【F:Gradio/run_pipeline.py†L44-L64】
2. **결합부위 분석** – ScanNet Flask API에서 ZIP 결과를 수신하고(`send_to_flask_server`), DeepBSRPred 스크립트를 로컬 실행한 뒤, 가중 평균과 연속 잔기 필터링을 통해 후보 영역을 도출합니다.【F:Gradio/function.py†L490-L536】【F:Gradio/function.py†L991-L1155】
3. **백본 생성** – RFdiffusion Docker 컨테이너에 override 플래그를 전달하여 다중 설계 구조를 생성하고, 결과를 컨티그/핫스팟 기준으로 정리합니다.【F:Gradio/run_pipeline.py†L67-L295】【F:Gradio/function.py†L117-L209】
4. **서열 설계** – ProteinMPNN 컨테이너로 서열을 샘플링하고, RMSD·pLDDT·확률 히트맵을 포함한 분석 결과를 생성합니다.【F:Gradio/run_pipeline.py†L322-L431】
5. **결합도 검증** – PRODIGY CLI 실행을 통해 도킹 구조의 결합 친화도를 계산하고, 3D 시각화와 요약 표를 제공합니다.【F:Gradio/function.py†L692-L913】

## 디렉토리 구성
- `Gradio/`: FastAPI·Gradio 앱, 파이프라인 제어 함수, 실행 스크립트 및 정적 자산이 위치합니다.【F:Gradio/app.py†L1-L502】【F:Gradio/function.py†L19-L1180】【F:Gradio/run_pipeline.py†L1-L431】
- `DeepBSRPred/`: DeepBSRPred 원본 스크립트와 의존 자원이 포함되어 있으며, Gradio 함수에서 해당 스크립트를 호출합니다.【F:Gradio/function.py†L1017-L1034】
- `ProteinMPNN/`: ProteinMPNN 앱과 Dockerfile이 포함되어 있으며, 컨테이너 내부에서 `app.py` UI를 재사용합니다.【F:ProteinMPNN/app.py†L1-L120】【F:ProteinMPNN/Dockerfile†L1-L39】
- `RFdiffusion/`: RFdiffusion 원본 코드와 Dockerfile, conda 환경 정의가 포함되어 있습니다.【F:RFdiffusion/Dockerfile†L1-L37】
- `ScanNet/`: ScanNet 관련 학습·추론 스크립트가 포함되어 있으며, 외부 Flask 서비스 구현 시 활용됩니다.【F:ScanNet/README.md†L1-L40】
- `dssp/`, `freesasa/`: DeepBSRPred가 필요로 하는 외부 도구 소스가 포함되어 있으며, DeepBSRPred 스크립트에서 해당 라이브러리를 직접 사용합니다.【F:DeepBSRPred/feature_calculation_prediction_ver1.py†L9-L37】

## 환경 준비
1. **Gradio 애플리케이션 의존성 설치**
   ```bash
   pip install -r Gradio/requirements.txt
   ```
   위 요구 사항 파일에는 FastAPI, Gradio, BioPython, Ray, Plotly, TensorFlow, PyTorch 등 전체 파이프라인 실행에 필요한 패키지가 정의되어 있습니다.【F:Gradio/requirements.txt†L1-L120】

2. **Docker 이미지 준비**
   - RFdiffusion: `RFdiffusion/Dockerfile`을 사용하여 CUDA 11.7 기반 이미지와 conda 환경을 구성합니다.【F:RFdiffusion/Dockerfile†L1-L37】
   - ProteinMPNN: `ProteinMPNN/Dockerfile`로 Python 환경과 애플리케이션을 설정합니다.【F:ProteinMPNN/Dockerfile†L1-L39】

3. **호스트 경로 및 서비스 엔드포인트 수정**
   - `Gradio/run_pipeline.py` 상단의 `HOST_*` 상수는 실제 배포 경로에 맞게 조정해야 합니다.【F:Gradio/run_pipeline.py†L18-L28】
   - ScanNet Flask API 주소(`FLASK_SERVER_URL`)를 환경에 맞게 수정하십시오.【F:Gradio/function.py†L490-L536】

4. **외부 도구 설치**
   DeepBSRPred 실행에는 DSSP, MSMS, HBPLUS, FreeSASA 등이 필요하므로 `/dssp`, `/freesasa` 등의 소스를 참고하여 사전 설치합니다.【F:DeepBSRPred/feature_calculation_prediction_ver1.py†L9-L37】

## 실행 방법
1. 필수 Docker 컨테이너(RFdiffusion, ProteinMPNN)를 가동 가능한 상태로 준비합니다.
2. ScanNet 예측을 위한 Flask 백엔드를 실행하거나, 제공된 엔드포인트를 이용합니다.【F:Gradio/function.py†L490-L536】
3. FastAPI 서버를 구동합니다.
   ```bash
   cd Gradio
   uvicorn app:app --host 0.0.0.0 --port 5001 --reload
   ```
   애플리케이션은 `/gradio` 경로로 Gradio UI를 마운트하며, `static/`과 `rfdiffusion_output/` 디렉터리가 FastAPI 정적 경로로 노출됩니다.【F:Gradio/app.py†L22-L24】【F:Gradio/app.py†L496-L502】
4. 브라우저에서 `http://<host>:5001/gradio`로 접속한 뒤, 탭을 순차적으로 실행하여 결합부위 탐색 → 백본 설계 → 서열 설계 → 도킹 검증을 진행합니다.【F:Gradio/app.py†L200-L497】

## 운영 팁
- RFdiffusion 출력은 컨티그/핫스팟 별 폴더로 정리되므로, 동일한 입력에 대한 반복 실험 시 해당 디렉터리를 삭제하거나 별도 보관하세요.【F:Gradio/function.py†L117-L209】
- ProteinMPNN 결과(`proteinmpnn_output/probs/*.npz`)는 후속 분석을 위해 CSV로 자동 변환되므로, 외부 분석 파이프라인과 쉽게 연계할 수 있습니다.【F:Gradio/run_pipeline.py†L372-L430】
- PRODIGY 분석 후에는 업로드한 ZIP 및 중간 PDB 파일이 자동 정리되지만, 실패 시 수동 삭제가 필요할 수 있습니다.【F:Gradio/function.py†L785-L807】
![image](https://github.com/user-attachments/assets/1b18a37e-0042-4831-a866-d82b1a3aa1c4)
## 다운방법
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

### ***출처 :
  https://github.com/RosettaCommons/RFdiffusion/tree/main
  
  https://github.com/ScanNet/ScanNet
  
  https://huggingface.co/spaces/simonduerr/ProteinMPNN

  https://web.iitm.ac.in/bioinfo2/deepbsrpred/download.html

## 참고
- RFdiffusion, ProteinMPNN, ScanNet, DeepBSRPred는 각각의 원저자 저장소 라이선스를 따르므로 사용 전에 해당 프로젝트의 정책을 확인하십시오.【F:RFdiffusion/README.md†L1-L80】【F:ProteinMPNN/README.md†L1-L60】【F:ScanNet/README.md†L1-L40】
