# 🎙️ Speaker Diarization Enhancement Using Denoising DL-Model
**딥러닝 기반 잡음 제거 모델을 활용한 화자 분리 정확도 향상 시스템**

## 🧩 프로젝트 데모
**딥러닝 기반 잡음 제거(denoising) + 듀얼 소스 VAD 하이브리드 모델을 통해 실제 교실/회의 환경에서도 DER을 낮춘 고정확도 화자 분리 시스템**

**딥러닝 기반 잡음 제거 이전 음성 파일**

https://github.com/user-attachments/assets/af81026d-572e-45cc-a3a0-27aeb1db8499

**딥러닝 기반 잡음 제거 후 음성 파일**

https://github.com/user-attachments/assets/1dcc7845-f84d-4b97-929a-7cb5d5a76158

**시연 영상**

https://github.com/user-attachments/assets/1f8c6f41-1cf3-4eeb-9b4e-60489110f855


## 🔗 기존 프로젝트 및 참고 링크
- **기존 프로젝트 GitHub:** [nemo-multistage-classroom-diarization](https://github.com/EduNLP/nemo-multistage-classroom-diarization.git)
- **Deep Learning Model GitHub:** [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)
- **참고 논문:** [EDM 2025 - Multistage Classroom Diarization](https://educationaldatamining.org/edm2025/proceedings/2025.EDM.short-papers.199/)  





## 📘 Overview  

### 🎯 문제점  
기존의 화자 분리 시스템은 **시끄럽고 다양한 소음이 존재하는 교실·회의 환경**에서 성능이 급격히 저하됨.  

### 💡 해결방안
본 프로젝트는 다음을 결합한 **다단계 하이브리드 파이프라인**을 제안함.  
1. **딥러닝 기반 Speech Enhancement (DeepFilterNet)**
   - 잡음 억제 + 화자 음색 보존  
2. **듀얼 소스 VAD (wav2vec2 + Whisper)**  
   - 소음 환경에서도 명확한 발화 구간 검출  
3. **NeMo 기반 Speaker Embedding + Clustering + Labeling**  
   - 깨끗한 오디오를 기반으로 화자 임베딩 품질 향상  
   - DER(Diarization Error Rate) 감소  

## 🧠 System Pipeline 
<img width="414" height="615" alt="image" src="https://github.com/user-attachments/assets/4c866d05-8407-42f0-9ecb-aa6ad0d880c3" />


---

## GUI
<img width="1920" height="1032" alt="GUI" src="https://github.com/user-attachments/assets/785e9d17-82bd-40a7-a0ed-277e3943320c" />

- Streamlit으로 제작된 전체적인 GUI



<img width="1374" height="105" alt="image" src="https://github.com/user-attachments/assets/a44495d6-8f61-49cc-bbdb-47e0fd6ea747" />

- 오디오 및 비디오 업로드



<img width="1390" height="109" alt="image" src="https://github.com/user-attachments/assets/387dc983-52f6-4caa-811f-9fcaa86bc8b3" />

- 입력 오디오 및 비디오로 Run diarization



<img width="248" height="653" alt="setting" src="https://github.com/user-attachments/assets/2f3eb6a6-d4cf-4f74-92a9-351d825f3782" />

- cuda 혹은 cpu 등의 running Device 선택 기능
- Denoise 모델 선택 기능
- Whisper 모델 선택 및 캐시 기능
- Speaker embedder 선택 기능
- 생성된 라벨링에 대해 하나의 문장으로 묶일 시간 설정 기능



<img width="250" height="370" alt="inference" src="https://github.com/user-attachments/assets/0f11bbe2-d4bc-4936-864e-69e635b664bf" />

- 화자 추론 방식 선택 기능
- 시나리오 선택 기능
- 구체적인 모델 선택 기능



<img width="1567" height="751" alt="script" src="https://github.com/user-attachments/assets/61cbffe2-a437-4296-b67b-fdbd73a51c54" />

- 최종 결과 Script



<img width="1190" height="85" alt="image" src="https://github.com/user-attachments/assets/1b63203e-5567-455c-abe6-04ed421134cf" />

- DER을 계산하기 위해 넣어둔 RTTM 존재 여부, 전체 파이프라인 사용과 관련은 없음 


---

## 📈 Result & Performance  

### 🧮 평가 지표 (DER)
```
DER = (FA + MISS + CER) / Duration
```
| Metric | 의미 |
|---------|------|
| FA (False Alarm) | 발화 없음 → 있음으로 오탐 |
| MISS | 실제 발화 → 미탐지 |
| CER (Confusion Error Rate) | 발화는 탐지했으나 화자 할당 오류 |


## Evaluation

- **평가 데이터 셋 GitHub:** (https://github.com/joonson/voxconverse?tab=readme-ov-file)


<img width="1800" height="1350" alt="dl_nodl_lines_2x2" src="https://github.com/user-attachments/assets/c9883c15-7e0a-4d2f-826d-6caddf8c8a11" />


- 테스트 데이터 셋에 대한 딥러닝 전처리 전후 비교



<img width="1650" height="1350" alt="dl_vs_nodl_scatter" src="https://github.com/user-attachments/assets/a5d0e531-c0f1-46b2-921a-cadf48305f7f" />


- 산점도 그래프


| ERROR 감소율 | Percentage |
|------|--------------|
| FA 감소율 | 0.36% |
| MISS 감소율 | -2.95% |
| CER 감소율 | 5.46% |
| DER 감소율 | 3.11% |



## ⚙️ Installation  
- OS: Windows 11
- GPU: RTX 3060

```
# git clone
git clone --branch feature/demucs-preprocessing --single-branch https://github.com/jth1097/Deep-learning-preproceng-pipelinssie-speaker-diarization-system.git

# powershell
# set virtual environment python version 3.11.4
python -m venv .venv
./.venv/Scripts/activate
python.exe -m pip install --upgrade pip

# dependency install
pip install -r requirements.txt
pip install -r requirements-gui.txt
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 # PyTorch 2.5.1 (CUDA 12.1 지원 버전)을 설치합니다.


# checkpoint
mkdir -p checkpoints/
gdown --fuzzy 'https://drive.google.com/file/d/1f9mMqzpGaLA2RB0m7dcesxo4deOB_GDq/view?usp=sharing' -O ckpt.pt
ren ckpt.pt w2v2.ckpt
mv w2v2.ckpt checkpoints
```

## 🚀 Usage  

```
# 실행
./run_gui.ps1
```

- `manifests/test.json` : 오디오 경로 및 메타데이터 목록  
- `vad_outs.json` : VAD 결과 중간 산출물  
- `diarization_output` : 최종 화자 분리 결과 저장 폴더  

## ⚠️ Common Issues & Solutions  

| 문제 | 원인 | 해결 방법 |
|------|------|------------|
| `PySoundFile failed` | libsndfile 미설치 | `sudo apt install libsndfile1` |
| CUDA 오류 (`device not found`) | GPU 환경 미설정 | CUDA 11.8 + cuDNN 8.6 버전 확인 |
| VAD 결과 없음 | 입력 파일 형식 불일치 | 16kHz mono PCM 형식으로 변환 |
| 0 bytes output | ffmpeg 변환 실패 | `ffmpeg -i input.wav -ar 16000 -ac 1 output.wav` 로 재생성 |
| 메모리 부족 | 딥러닝 모델 메모리 초과 | `--batch_size` 감소 또는 GPU 메모리 증가 필요 |


## 🧩 Future Work  
- 딥러닝 전처리 모델 **Fine-tuning** (잡음 포함 vs 제거 데이터 병합 학습)  
- 실시간 스트리밍 환경 적용 (on-device inference 최적화)


## 👥 Team “Alone”
| 역할 | 이름 |
|------|------|
| Researcher |  신홍규 |
| Researcher |  남경식 |
| Researcher |  양평화 |
| Researcher |  장태환 |



## 🧾 License  
This project is for **academic research** purposes under the **Konkuk University Capstone Design (졸업프로젝트)** program.  
For any citation or reuse, please credit:  
> *ALONE et al., “Speaker diarization enhancement using denoising DL-model”, Konkuk Univ., 2025.*
