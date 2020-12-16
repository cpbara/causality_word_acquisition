echo basic
python3 test_full_action_classifier.py --pretrained_model=models/basic_causal_action_classifier_70.torch 
python3 test_full_action_classifier.py --pretrained_model=models/basic_causal_action_classifier_30.torch 
python3 test_full_action_classifier.py --pretrained_model=models/basic_causal_action_classifier_20.torch 
python3 test_full_action_classifier.py --pretrained_model=models/basic_causal_action_classifier_10.torch 
echo predicted bboxes
python3 test_full_action_classifier.py --pretrained_model=models/pr_bbox_causal_action_classifier_70.torch --bbox_type='pr_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/pr_bbox_causal_action_classifier_30.torch --bbox_type='pr_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/pr_bbox_causal_action_classifier_20.torch --bbox_type='pr_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/pr_bbox_causal_action_classifier_10.torch --bbox_type='pr_bbox'
echo gt bboxes
python3 test_full_action_classifier.py --pretrained_model=models/gt_bbox_causal_action_classifier_70.torch --bbox_type='gt_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/gt_bbox_causal_action_classifier_30.torch --bbox_type='gt_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/gt_bbox_causal_action_classifier_20.torch --bbox_type='gt_bbox'
python3 test_full_action_classifier.py --pretrained_model=models/gt_bbox_causal_action_classifier_10.torch --bbox_type='gt_bbox'
echo mental attention
python3 test_full_action_classifier.py --pretrained_model=models/mental_attention_action_classifier_70.torch --attention_model=models/mental_attention_model.torch
python3 test_full_action_classifier.py --pretrained_model=models/mental_attention_action_classifier_30.torch --attention_model=models/mental_attention_model.torch
python3 test_full_action_classifier.py --pretrained_model=models/mental_attention_action_classifier_20.torch --attention_model=models/mental_attention_model.torch
python3 test_full_action_classifier.py --pretrained_model=models/mental_attention_action_classifier_10.torch --attention_model=models/mental_attention_model.torch
