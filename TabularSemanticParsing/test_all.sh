./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token --db_dir data/ko_token --beam_size 4
./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token_not_h --db_dir data/ko_token_not_h --beam_size 4
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table --db_dir data/ko_from_table --beam_size 4
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table_not_h --db_dir data/ko_from_table_not_h --beam_size 4

./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token --db_dir data/ko_token --beam_size 8
./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token_not_h --db_dir data/ko_token_not_h --beam_size 8
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table --db_dir data/ko_from_table --beam_size 8
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table_not_h --db_dir data/ko_from_table_not_h --beam_size 8

./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token --db_dir data/ko_token --beam_size 16
./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token_not_h --db_dir data/ko_token_not_h --beam_size 16
./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table --db_dir data/ko_from_table --beam_size 16
./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table_not_h --db_dir data/ko_from_table_not_h --beam_size 16

# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token --db_dir data/ko_token --beam_size 64
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_token_not_h --db_dir data/ko_token_not_h --beam_size 64
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table --db_dir data/ko_from_table --beam_size 64
# ./experiment-bridge.sh configs/bridge/wikisql-bridge-bert-large.sh --inference 0 --test --data_dir data/ko_from_table_not_h --db_dir data/ko_from_table_not_h --beam_size 64
