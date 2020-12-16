TRAIN_LIST="/scratch/chaijy_root/chaijy1/cpbara/actions-transitions/labels/task1/trainlist.txt"
TEST_LIST="/scratch/chaijy_root/chaijy1/cpbara/actions-transitions/labels/task1/testlist.txt"
DATA_ROOT_PATH="/scratch/chaijy_root/chaijy1/cpbara/actions-transitions/action_frames_release/"
echo test basic action classifier
python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_5.torch --train_set_size=5 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
#python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_30.torch --train_set_size=30 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
#python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_70.torch --train_set_size=70 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
echo test action classifier with mental attention network
python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_5.torch --train_set_size=5 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
#python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_30.torch --train_set_size=30 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
#python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_70.torch --train_set_size=70 --train_list=${TRAIN_LIST} --test_list=${TEST_LIST} --data_root_path=${DATA_ROOT_PATH}
