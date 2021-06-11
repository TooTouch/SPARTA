# ===============
# 방법 1. ko_token: header 번역 x
# 방법 2. ko_token_not_h: header 번역 
# 방법 3. ko_from_table: from table 
# 방법 4. ko_from_table_not_h: from table 
# ===============

input=$1
dataname="ko_token ko_token_not_h ko_from_table ko_from_table_not_h"

echo "[TRANSLATION] replace value with token"
python translate.py --replace $input --datadir ./data/data --savedir ./data/ko_token --header
echo "[TRANSLATION] Complete ko_token"
python translate.py --replace $input --datadir ./data/data --savedir ./data/ko_token_not_h
echo "[TRANSLATION] Complete ko_token_not_h"
python translate.py --replace $input --datadir ./data/data --savedir ./data/ko_from_table --from_table --header
echo "[TRANSLATION] Complete ko_from_table"
python translate.py --replace $input --datadir ./data/data --savedir ./data/ko_from_table_not_h --from_table
echo "[TRANSLATION] Complete ko_from_table_not_h"
