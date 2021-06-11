# Make Folder 
echo "[FOLDER]"

savedir="data"

if [ ! -d $savedir ];
then
	mkdir $savedir
	echo "Folder Created"
else
	echo "Folder exists"
fi

echo Change Working Directory to $savedir
cd ./$savedir

echo "[DOWNLOAD]"

# Data Download Using Kaggle API
kaggle datasets download -d ananta/credit-card-data -f CardBase.csv   
kaggle datasets download -d ananta/credit-card-data -f CustomerBase.csv  
kaggle datasets download -d darpan25bajaj/credit-card-exploratory-data-analysis -f "Customer Acqusition.csv"  
kaggle datasets download -d darpan25bajaj/credit-card-exploratory-data-analysis -f Repayment.csv  
kaggle datasets download -d darpan25bajaj/credit-card-exploratory-data-analysis -f spend.csv  
kaggle datasets download -d ananta/credit-card-data -f TransactionBase.csv   
kaggle datasets download -d rikdifos/credit-card-approval-prediction -f application_record.csv  
unzip application_record.csv.zip
rm *.zip
kaggle datasets download -d bukolafatunde/personal-finance -f Budget.csv  
kaggle datasets download -d bukolafatunde/personal-finance -f personal_transactions.csv   
kaggle datasets download -d janiobachmann/bank-marketing-dataset -f bank.csv 

datanames=("CardBase.csv" "CustomerBase.csv" "Customer%20Acqusition.csv" "Repayment.csv" "spend.csv" \
	   "TransactionBase.csv" "application_record.csv" "Budget.csv" "personal_transactions.csv" "bank.csv")
# table_ids=("1-100000-1" "1-100000-2" "1-100000-3" "1-100000-4" "1-100000-5" \
# 	   "1-100000-6" "1-100000-7" "1-100000-8" "1-100000-9" "1-100000-10")
table_ids=("CardBase" "CustomerBase" "CustomerAcqusition" "CustomerRepayment" "CustomerSpend" \
           "TransactionBase" "ApplicationRecord" "Budget" "PersonalTransaction" "Bank")

total_len=${#table_ids[@]}

for ((i=0;i<$total_len;i++));
do
	mv ${datanames[i]} ${table_ids[i]}.csv
done

echo "[CREATE]"
cd ..
python kaggle_demo.py --datadir data

cp kaggle_demo_question.jsonl data/test.jsonl
