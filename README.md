# BindQ
![image](https://github.com/user-attachments/assets/1b18a37e-0042-4831-a866-d82b1a3aa1c4)
## 다운방법
### 1. 모든 디렉토리를 내려받기
- ProteinMPNN 하위에 params 디렉토리를 생성하고 wget으로 파라미터를 다운로드받는다.

  wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/LICENSE
  
  wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_1.npz
  
  wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_1_ptm.npz
  
  wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_5.npz
  
  wget https://huggingface.co/spaces/simonduerr/ProteinMPNN/resolve/main/params/params_model_5_ptm.npz

- RFdiffusion 하위에 models 디렉토리를 생성하고 wget으로 모델을 다운로드 받는다.

   wget http://files.ipd.uw.edu/pub/RFdiffusion/6f5902ac237024bdd0c176cb93063dc4/Base_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/e29311f6f1bf1af907f9ef9f44b8328b/Complex_base_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/60f09a193fb5e5ccdc4980417708dbab/Complex_Fold_base_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/74f51cfb8b440f50d70878e05361d8f0/InpaintSeq_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/76d00716416567174cdb7ca96e208296/InpaintSeq_Fold_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/5532d2e1f3a4738decd58b19d633b3c3/ActiveSite_ckpt.pt
 
   wget http://files.ipd.uw.edu/pub/RFdiffusion/12fc204edeae5b57713c5ad7dcb97d39/Base_epoch8_ckpt.pt

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

### ***기타 :
  해당 소스코드는 향후 리팩토링예정
