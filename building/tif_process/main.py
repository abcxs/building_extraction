# -*- coding: utf-8 -*-

import argparse
import config as cfg
import gdal
import multiprocessing
import os
import pickle
import subprocess
import sys
import time
import torch
from detect import detect
from gen_shp import gen_shp
from post_process import post_process
from utils import cut_into_pieces, get_gpus, logger

# os.environ['BUILDING_CUDA'] = '0,1,2,3,4,5,6,7'

def parse_args():
    parser = argparse.ArgumentParser(description='Building extraction')
    parser.add_argument(
        'input',
        help='input dir'
    )
    parser.add_argument(
        'output',
        help='output dir'
    )
    parser.add_argument(
        'config',
        help='config file path'
    )
    parser.add_argument(
        'checkpoint',
        help='checkpoint file',
    )
    parser.add_argument('--approx_polygon', action='store_true')
    # params = ['/home/ndrcchkygb/data/sample',
    #           '/home/ndrcchkygb/project/temp_result/t5',
    #           '/home/ndrcchkygb/code/mmdetection/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_building_base_fp_background_ms.py',
    #           '/home/ndrcchkygb/code/mmdetection/work_dirs/mask_rcnn_r50_fpn_1x_building_base_fp_background_ms/epoch_12.pth',
    #           '--approx_polygon'
    #           ]
    return parser.parse_args()


def deal_with_single_tif(tif_file, output_dir, config_file, checkpoint, approx_polygon):
    gpus = get_gpus()
    gpus_info = [str(id_) for id_ in gpus]
    logger.info('使用GPU:%s' % ','.join(gpus_info))

    process_per_gpu = cfg.process_per_gpu
    num_of_process = len(gpus) * process_per_gpu

    piece_list = cut_into_pieces(tif_file, cfg)

    if num_of_process == 0:
        detect(0, gpus[0], config_file, checkpoint, tif_file,
               piece_list, output_dir, approx_polygon)
    else:
        processes = []
        for i in range(num_of_process):
            gpu_id = gpus[i % len(gpus)]
            piece_list_per_process = piece_list[i::num_of_process]
            if len(piece_list_per_process) <= 0:
                continue
            ctx = multiprocessing.get_context('spawn')
            p = ctx.Process(target=detect, args=(
                i, gpu_id, config_file, checkpoint, tif_file, piece_list_per_process, output_dir, approx_polygon))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    logger.info('all detect process quit')
    detections = []
    for i in range(num_of_process):
        if not os.path.exists(os.path.join(output_dir, 'detection_%d.pkl' % i)):
            continue
        with open(os.path.join(output_dir, 'detection_%d.pkl' % i), 'rb') as f:
            detections.extend(pickle.load(f))
    with open(os.path.join(output_dir, 'detection.pkl'), 'wb') as f:
        pickle.dump(detections, f)
    for i in range(num_of_process):
        if not os.path.exists(os.path.join(output_dir, 'detection_%d.pkl' % i)):
            continue
        os.remove(os.path.join(output_dir, 'detection_%d.pkl' % i))


if __name__ == "__main__":
    args = parse_args()
    input_dir = args.input
    config_file = args.config
    checkpoint = args.checkpoint

    if not os.path.exists(input_dir):
        logger.info('目录不存在')
        sys.exit()

    tif_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.endswith('.tif'):
                tif_file = os.path.join(root, f)
                tif_files.append(tif_file)

    input_dir = os.path.dirname(input_dir)
    for tif_file in tif_files:
        t1 = time.time()

        file_name = os.path.splitext(os.path.basename(tif_file))[0]
        output_dir = os.path.join(
            root, file_name).replace(input_dir, args.output)

        logger.info(f'开始处理 {tif_file}')

        if os.path.exists(os.path.join(output_dir, 'flag')):
            logger.info(f'{tif_file} 已经被处理过，跳过')
            continue

        ds = gdal.Open(tif_file, gdal.GA_ReadOnly)
        if ds is None or ds.GetGeoTransform() is None:
            logger.info(f'{tif_file} 打开失败，跳过')
            continue

        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(os.path.join(output_dir, 'detection.pkl')):
            logger.info(f'开始检测 {tif_file}')
            deal_with_single_tif(tif_file, output_dir, config_file, checkpoint, args.approx_polygon)

        logger.info('开始后处理')
        width = ds.RasterXSize
        height = ds.RasterYSize

        post_process(output_dir, cfg, width, height)
        # p = multiprocessing.Process(
        #     target=post_process, args=(output_dir, cfg, width, height))
        # p.start()
        # p.join()

        logger.info('生成shp文件')
        pkl_files = ['poly', 'mask', 'boxes']
        for pkl_file in pkl_files:
            input_pkl = os.path.join(output_dir, '%s.pkl' % pkl_file)
            output_file = os.path.join(
                output_dir, '%s_%s.shp' % (file_name, pkl_file))
            gen_shp(ds, input_pkl, output_file)

        logger.info('生产结束文件')
        with open(os.path.join(output_dir, 'flag'), 'w') as f:
            f.write('')

        logger.info('删除多余文件')
        for pkl_file in pkl_files:
            os.remove(os.path.join(output_dir, '%s.pkl' % pkl_file))
        os.remove(os.path.join(output_dir, 'detection.pkl'))
        logger.info(f'{tif_file} 处理完成')

        t2 = time.time()
        total_time = int(t2 - t1)
        m, s = divmod(total_time, 60)
        h, m = divmod(m, 60)
        logger.info(f'处理 {tif_file} 花费 {h} 小时 {m} 分钟 {s} 秒.')
