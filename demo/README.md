# Demo 

한국어 wikisql 검증을 위한 demo 

# Demo DB

- Kaggle Data

Table ID | Name
---|---
1-100000-1 |  [CardBase](https://www.kaggle.com/ananta/credit-card-data?select=CardBase.csv)
1-100000-2 |  [CustomerBase](https://www.kaggle.com/ananta/credit-card-data?select=CustomerBase.csv)
1-100000-3 |  [Customer Acqusition](https://www.kaggle.com/darpan25bajaj/credit-card-exploratory-data-analysis?select=Customer+Acqusition.csv)
1-100000-4 |  [Repayment](https://www.kaggle.com/darpan25bajaj/credit-card-exploratory-data-analysis?select=Repayment.csv)
1-100000-5 |  [Spend](https://www.kaggle.com/darpan25bajaj/credit-card-exploratory-data-analysis?select=spend.csv)
1-100000-6 |  [TransactionBase](https://www.kaggle.com/ananta/credit-card-data?select=TransactionBase.csv)
1-100000-7 |  [Application Record](https://www.kaggle.com/rikdifos/credit-card-approval-prediction?select=application_record.csv)
1-100000-8 |  [Budget](https://www.kaggle.com/bukolafatunde/personal-finance?select=Budget.csv)
1-100000-9 |  [Personal Transactions](https://www.kaggle.com/bukolafatunde/personal-finance?select=personal_transactions.csv)
1-100000-10 | [Bank](https://www.kaggle.com/janiobachmann/bank-marketing-dataset?select=bank.csv)

# File Description

```bash
demo
├── README.md
└── kaggle_demo 
    ├── kaggle_demo.py # database 및 table schema 생성
    ├── kaggle_demo_question.jsonl # 검증 질문 및 정답 SQL
    └── run.sh # Kaggle API를 활용하여 데이터 다운로드 및 kaggle_demo.py 실행
```

# How to Build DB

## 1. Kaggle API 

kaggle data를 한번에 다운 받기 위해서 **kaggle API**를 사용합니다. **kaggle API**를 사용하기 위한 과정을 아래 절차를 따르시고 보다 자세한 내용은 [여기](https://github.com/Kaggle/kaggle-api)를 참고하세요.

1. pip를 통해 `kaggle` 설치

```bash
pip install kaggle
```

2. https://www.kaggle.com 에 접속하여 로그인
3. 우측 상단 사용자의 아이콘 클릭 후 Account 접속
4. `Create New API Token`을 눌러서 `kaggle.json` 다운로드
5. 다운도르 폴더 위치에 아래 명령어를 통해 권한 변경

```bash
chmod 600 kaggle.json
```

6. 아래 명령어를 통해 환경변수 설정

```bash
export KAGGLE_USERNAME=datadinosaur
export KAGGLE_KEY=xxxxxxxxxxxxxx
```

## 2. Run

아래 명령어는 크게 세 가지 과정을 거칩니다.

1. Kaggle 데이터 다운로드
2. `sqlite3`를 사용하여 `test.db`와 `test.tables.jsonl` 파일 생성
3. 검증을 위해 작성한 질문과 정답 SQL이 들어있는 `kaggle_demo_question.jsonl`을 `test.jsonl`로 변경하여 생성된 data 폴더에 저장

```bash
bash kaggle_demo/run.sh
```
