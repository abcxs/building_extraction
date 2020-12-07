import os

site = 'gansu'
base_dir = '/home/ndrcchkygb/data/building/dst/%s' % site
gt_dir = os.path.join(base_dir, 'eval', 'gt')
prediction_dir = os.path.join(base_dir, 'eval', 'prediction')
filelist_json = os.path.join(base_dir, 'filelist.json')

exclude_dir = []
threshold = 0.15
eval_type = 'boxes'

ovthresh = 0.05
