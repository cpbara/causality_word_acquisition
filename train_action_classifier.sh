echo basic action classifier
python train_action_classifier.py --out_model_path=models/baseline_action_classifier_5.torch --train_set_size=5
python train_action_classifier.py --out_model_path=models/baseline_action_classifier_30.torch --train_set_size=30
python train_action_classifier.py --out_model_path=models/baseline_action_classifier_70.torch --train_set_size=70
echo action classifier with mental attention network
python train_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/baseline_action_classifier_5.torch --out_model_path=models/ma_action_classifier_5.torch --train_set_size=5
python train_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/baseline_action_classifier_30.torch --out_model_path=models/ma_action_classifier_30.torch --train_set_size=30
python train_action_classifier.py --ma_emb_sufix=MentAttFromNounGrounding --pretrained_model=models/baseline_action_classifier_70.torch --out_model_path=models/ma_action_classifier_70.torch --train_set_size=70
