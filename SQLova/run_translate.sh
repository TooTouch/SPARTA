# ===============
# 방법 1. ko_token: header 번역 x
# ===============
# python translate.py --replace value --datadir ./data/fix_raw --savedir ./data/ko_token --header
python translate.py --replace token --datadir ./data/fix_raw --savedir ./data/ko_token --header
# ===============
# 방법 2. ko_token_not_h: header 번역 
# ===============
# python translate.py --replace value --datadir ./data/fix_raw --savedir ./data/ko_token_not_h
python translate.py --replace token --datadir ./data/fix_raw --savedir ./data/ko_token_not_h
# ===============
# 방법 3. ko_from_table: from table 
# ===============
# python translate.py --replace value --from_table --header --datadir ./data/fix_raw --savedir ./data/ko_from_table
python translate.py --replace token --from_table --header --datadir ./data/fix_raw --savedir ./data/ko_from_table
# ===============
# 방법 4. ko_from_table_not_h: from table 
# ===============
# python translate.py --replace value --from_table --datadir ./data/fix_raw --savedir ./data/ko_from_table_not_h
python translate.py --replace token --from_table --datadir ./data/fix_raw --savedir ./data/ko_from_table_not_h