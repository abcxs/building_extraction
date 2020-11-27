import config as cfg
import os


def summery(input_dir, threshold):
    detect_types = ['poly', 'mask', 'boxes']
    for detect_type in detect_types:
        txt_files = []
        for root, _, files in os.walk(os.path.join(input_dir, 'gt')):
            for f in files:
                if f == '%s.txt' % detect_type:
                    txt_files.append(os.path.join(root, f))
        result = []
        for txt_f in txt_files:
            fn = os.path.basename(os.path.dirname(txt_f))
            with open(txt_f) as f:
                ins = f.read().split('\n')
            ins = [obj.strip().split(' ') for obj in ins if obj]
            for obj in ins:
                score = float(obj[1])
                if score > threshold:
                    result.append(' '.join([fn] + obj[1:]))
        with open(os.path.join(input_dir, '%s.txt' % detect_type), 'w') as f:
            f.write('\n'.join(result))

summery(cfg.prediction_dir, cfg.threshold)
