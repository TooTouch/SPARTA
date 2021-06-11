unzip data.zip

if [[ -d ../data/ko_from_table ]]
then
    echo "../data/ko_from_table exists"
    mv data/wikisql.bridge.question-split.ppl-0.85.2.dn.no_from.bert.multilingual.data.pkl ../data/ko_from_table/
else
    echo "../data/ko_from_table doesn't exists"
    echo "Commencing Unzipping the dataset"
    
    unzip ../data/ko_from_table.zip -d ../data/
    mv data/wikisql.bridge.question-split.ppl-0.85.2.dn.no_from.bert.multilingual.data.pkl ../data/ko_from_table/
fi

