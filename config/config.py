ROOT = '../my_data'
dataset_configs = {
    'imgs_path': ROOT+'/images/images',
    'annotations': ROOT + '/df.csv',
    'normalize': {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    'candidate_boxes_path': ROOT + '/images/candidates',
    'candidate_boxes_class': ROOT + '/images/classes',
    'candidate_boxes_delta': ROOT + '/images/delta',
    'num_workers': 8,
    'IoU_threshold': 0.35,
    'train_ratio': 0.7,
    'test_ratio': 0.15,
    'weight_decay': 0.0005,
}
