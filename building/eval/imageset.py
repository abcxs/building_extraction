
import config as cfg
import glob
import json
import os

val_txts = glob.glob(os.path.join(cfg.gt_dir, '*.txt'))
file_names = [os.path.splitext(os.path.basename(val_txt))[0] for val_txt in val_txts]
file_names = [file_name for file_name in file_names if file_name not in cfg.exclude_dir]

with open(cfg.filelist_json) as f:
    files = json.load(f)
id2name = {}
for k, v in files.items():
    if v['split'] == 'val':
        id2name[v['id']] = os.path.basename(k)

file_names = ['%s %s' % (file_name, id2name[int(file_name.split('_')[0])]) for file_name in file_names]
with open(os.path.join(cfg.base_dir, 'eval', 'imageset.txt'), 'w') as f:
    f.write('\n'.join(file_names))
