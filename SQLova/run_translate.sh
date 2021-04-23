# ===============
# ko_token: header 번역 x
# ===============
python translate.py --replace value --savedir ./data/ko_token --header
python translate.py --replace token --savedir ./data/ko_token --header
# ===============
# ko_token_not_h: header 번역 
# ===============
python translate.py --replace value --savedir ./data/ko_token_not_h
python translate.py --replace token --savedir ./data/ko_token_not_h
# ===============
# ko_from_table: from table 
# ===============
python translate.py --replace value --from_table --savedir ./data/ko_from_table
python translate.py --replace token --from_table  --savedir ./data/ko_from_table
