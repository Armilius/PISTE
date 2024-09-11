## training data using ramdon sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_training_data.csv --neg ../data/raw_data/neg_training_data.csv --sampling shuffle --file_type training --neg_ratio 10

## training data using unipep sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_training_data.csv --neg ../data/raw_data/neg_training_data.csv --sampling unipep --file_type training --neg_ratio 10

## training data using reftcr sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_training_data.csv --neg ../data/raw_data/neg_training_data.csv --sampling reftcr --ref_tcr ../data/raw_data/ref_healthy_tcr.csv --file_type training --neg_ratio 10

## testdata-1 using random sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_1.csv --neg ../data/raw_data/neg_test_1.csv --sampling shuffle --file_type test --neg_ratio 10

## testdata-1 using unipep sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_1.csv --neg ../data/raw_data/neg_test_1.csv --sampling unipep --file_type test --neg_ratio 10

## testdata-1 using reftcr sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_1.csv --neg ../data/raw_data/neg_test_1.csv --sampling reftcr --ref_tcr ../data/raw_data/ref_tcr_test1.csv --file_type test --neg_ratio 10


## testdata-2 using random sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_2.csv --neg ../data/raw_data/neg_test_2.csv --sampling shuffle --file_type dbpepneo --neg_ratio 10

## testdata-2 using unipep sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_2.csv --neg ../data/raw_data/neg_test_2.csv --sampling unipep --file_type dbpepneo --neg_ratio 10

## testdata-2 using reftcr sampling
python ./Dataset/sampler.py --pos ../data/raw_data/pos_test_2.csv --neg ../data/raw_data/neg_test_2.csv --sampling reftcr --ref_tcr ../data/raw_data/ref_tcr_test2.csv --file_type dbpepneo --neg_ratio 10

