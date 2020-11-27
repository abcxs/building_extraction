import config as cfg
import glob
import json
import os
import shutil

data_dir = os.path.join(cfg.base_dir, 'tmp')
filelist_json = cfg.filelist_json
gt_dir = cfg.gt_dir
os.makedirs(gt_dir, exist_ok=True)

with open(filelist_json) as f:
    files = json.load(f)
val_ids = []
for k, v in files.items():
    if v['split'] == 'val':
        val_ids.append(v['id'])
val_ids = set(val_ids)
data_txts = glob.glob(os.path.join(data_dir, '*.txt'))
for txt_file in data_txts:
    fn = os.path.basename(txt_file)
    id_ = int(fn.split('_')[0])
    if id_ in val_ids:
        shutil.copyfile(txt_file, os.path.join(gt_dir, fn))
        shutil.copyfile(txt_file.replace('txt', 'png'), os.path.join(gt_dir, fn.replace('txt', 'png')))

