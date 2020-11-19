# 将shp文件转为txt

import gdal
import glob
import json
import numpy as np
import ogr
import os
import osr
import shutil
from utils import area_of_boxes, cr2geo

min_box_size = 10

def check_mask(tif_files):
    boxes = []
    for tif_file in tif_files:
        dataset = gdal.Open(tif_file)
        geoTransform = dataset.GetGeoTransform()
        width = dataset.RasterXSize
        height = dataset.RasterYSize
        pos = cr2geo(geoTransform, width, height)
        boxes.append([geoTransform[0], -geoTransform[3], pos[0], -pos[1]])
    boxes = np.array(boxes)
    inter_area = area_of_boxes(boxes, boxes)
    for i in range(boxes.shape[0]):
        inter_area[i, i] = 0
    mask = np.any(inter_area, axis=0)
    return mask.tolist()


def shp2txt(tifFile, shpFile, is_need_mask):
    ogr.RegisterAll()

    gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")

    dataset = gdal.Open(tifFile)
    width = dataset.RasterXSize
    height = dataset.RasterYSize
    geoTransform = dataset.GetGeoTransform()

    srcArray = dataset.ReadAsArray()
    if srcArray is None:
        print('read error, just skip')
        return []
    else:
        srcArray = srcArray.astype(np.uint8)

    dataSource = ogr.Open(shpFile)
    if dataSource is None:
        print('fail to open', shpFile)
        return []
    daLayer = dataSource.GetLayer(0)
    featureCount = daLayer.GetFeatureCount()

    prosrs = daLayer.GetSpatialRef()
    if prosrs is None:
        print('the coordinate system cannot be determined, just skip')
        return []
    geosrs = osr.SpatialReference()
    geosrs.SetWellKnownGeogCS("WGS84")
    ct = osr.CoordinateTransformation(prosrs, geosrs)

    msgs = []

    for _ in range(featureCount):
        feature = daLayer.GetNextFeature()

        geometry = feature.GetGeometryRef()
        if geometry == None:
            continue

        ring = geometry.GetGeometryRef(0)
        numPoints = ring.GetPointCount()
        if numPoints < 3:
            continue

        msg = ['building']

        points = []
        max_y = 0
        max_x = 0
        min_y = height - 1
        min_x = width - 1
        for j in range(numPoints - 1):
            coords = ct.TransformPoint(ring.GetX(j), ring.GetY(j))[:2]
            dcol, drow = geo2cr(geoTransform, coords[0], coords[1])

            dcol = max(min(dcol, width - 1), 0)
            drow = max(min(drow, height - 1), 0)
            msg.append('%.2f' % dcol)
            msg.append('%.2f' % drow)
            points.append([dcol, drow])

            max_x = max(max_x, dcol)
            max_y = max(max_y, drow)
            min_x = min(min_x, dcol)
            min_y = min(min_y, drow)

        if msg and max_y - min_y > min_box_size and max_x - min_x > min_box_size:
            if is_need_mask:
                mask = np.zeros((height, width, 3), dtype=np.uint8)
                points = np.array(points).reshape(-1, 1, 2).astype(np.int)
                cv2.fillPoly(mask, [points], (255, 255, 255))
                mask = cv2.bitwise_and(mask.transpose(2, 0, 1), srcArray)
                if mask.any():
                    msgs.append(' '.join(msg))
            else:
                msgs.append(' '.join(msg))
    return msgs

def batshp2txt(files, output_dir):
    for f in files:
        id_ = f['id']
        tif_files = f['tif']
        shp_files = f['shp']
        mask = check_mask(tif_files)
        for tif_file, is_need_mask in zip(tif_files, mask):
            file_name = os.path.splitext(os.path.basename(tif_file))[0]
            for shp_file in shp_files:
                print(is_need_mask, id_, tif_file, shp_file)
                msgs = shp2txt(tif_file, shp_file, is_need_mask)
                if len(msgs) > 0:
                    with open(os.path.join(output_dir, '%s_%d.txt' % (file_name, id_)), 'w') as f:
                        f.write('\n'.join(msgs))
                    shutil.copyfile(tif_file, os.path.join(output_dir, '%s_%d.tif' % (file_name, id_)))



# %%
site = 'hubei'
val_dir = '/data/zfp/data/building/%s/val/tmp' % site
input_json = '/data/zfp/data/building/%s/filelist.json' % site
output_dir = '/data/building_data/eval/%s/data' % site
os.makedirs(output_dir, exist_ok=True)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(input_json) as f:
    files = json.load(f)['data']
val_txts = glob.glob(os.path.join(val_dir, '*.txt'))
val_ids = [int(os.path.basename(val_txt).split('_')[0]) for val_txt in val_txts]
val_files = []
for file in files:
    id_ = file['id']
    tifs = file['tif']
    shps = file['shp']
    if id_ in val_ids:
        val_files.append(file)

batshp2txt(val_files, output_dir)

