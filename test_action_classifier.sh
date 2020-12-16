echo test basic action classifier
python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_5.torch --train_set_size=5
python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_30.torch --train_set_size=30
python test_action_classifier.py --pretrained_model=models/baseline_action_classifier_70.torch --train_set_size=70
echo test action classifier with mental attention network
python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_5.torch --train_set_size=5
python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_30.torch --train_set_size=30
python test_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/ma_action_classifier_70.torch --train_set_size=70
