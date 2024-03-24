# 도배 하자 질의 응답 처리 : 한솔데코 시즌2 AI 경진대회
**2024.01.29 ~ 2024.03.11**  
[**Competition Page**](https://dacon.io/competitions/official/236216/overview/description)  
![image](https://github.com/KU-BIG/KUBIG_2024_SPRING/assets/104672441/1694a3fe-da66-46af-93c9-efdd070010d0)
# Project Structure
```
Hansol-QA/
│
├── configs/ - configuration file for model train and inference
│   ├── datavortex_*.json
│   └── ldsolar_*.json
│
├── experiments/ - experimental ipynb files for model and technique application
│   ├── EDA.ipynb
│   ├── config_crypto.ipynb: encrypt configuration information
│   ├── klue_Roberta-large.ipynb: predict category for test data question
│   ├── papago_backtranslation.ipynb: backtranlation data augmentation with papago api
│   ├── question_similarity_check.ipynb: measure the similarity between Q1 and Q2 and select the Q2 to exclude
│   ├── rag_chromadb.ipynb: RAG Techniques Using Chromadb
│   ├── test_split.ipynb: dividie test data by conjunction
│   └── trained_inference_test.ipynb: qualitative assessment of the inference ability of the trained model
│
├── modules/ - functions and classes required to operate the model
│   ├── dataloader.py
│   ├── trainer.py
│   └── utils.py
│
├── templates/ - template for creating qa dataset
│   ├── datavortex.txt
│   └── ldcc.txt
│
├── requirements.txt - requirements for carrying out the project
├── train.py - main script to start training
└── inference.py - make submission with trained models
```
# Models & References
- Models
  - [kogpt2-base-v2](https://huggingface.co/skt/kogpt2-base-v2)
  - [OPEN-SOLAR-KO-10.7B](https://huggingface.co/beomi/OPEN-SOLAR-KO-10.7B)
  - [LDCC-SOLAR-10.7B](https://huggingface.co/LDCC/LDCC-SOLAR-10.7B)
  - [DataVortexS-10.7B-dpo-v1.11](https://huggingface.co/Edentns/DataVortexS-10.7B-dpo-v1.11)
  - [roberta-large](https://huggingface.co/klue/roberta-large)
- References
  - [Transformers Library](https://huggingface.co/docs/transformers/index)
  - [Langchain, ChromaDB](https://python.langchain.com/docs/integrations/vectorstores/chroma)
  - [open-ko-llm-leaderboard](https://huggingface.co/spaces/upstage/open-ko-llm-leaderboard)
