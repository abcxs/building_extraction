try:
    import gdal
except ImportError:
    from osgeo import gdal
try:
    import ogr
except ImportError:
    from osgeo import ogr
try:
    import osr
except ImportError:
    from osgeo import osr

import os
import pickle
from PIL import Image
from utils import logger


def transformer(geoTransform, x, y):
    xx = geoTransform[0] + x * geoTransform[1] + y * geoTransform[2]
    yy = geoTransform[3] + x * geoTransform[4] + y * geoTransform[5]

    return xx, yy


def gen_shp(ds, input_pkl, output_file, finsh_flag=None):
    ds_image = False
    if isinstance(ds, Image.Image):
        geoTransform = (0, 1, 0, 0, 0, -1)
        ds_image = True
    else:
        geoTransform = ds.GetGeoTransform()
    gdal.SetConfigOption('SHAPE_ENCODING', "UTF8")
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    ogr.RegisterAll()
    driverName = "ESRI Shapefile"
    driver = ogr.GetDriverByName(driverName)
    if driver is None:
        logger.info(f"1 {driverName} driver failed!")
    datasource = driver.CreateDataSource(output_file)
    if datasource is None:
        logger.info(f"2 {output_file} create failed!")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    if not ds_image:
        prosrs = osr.SpatialReference()
        prosrs.ImportFromWkt(ds.GetProjection())
        ct = osr.CoordinateTransformation(prosrs, srs)
    
    layer = datasource.CreateLayer("BuildPolygon", srs, ogr.wkbPolygon)
    if layer is None:
        logger.info("3 failed!")
    filed_name = ogr.FieldDefn("Confidence", ogr.OFTReal)
    layer.CreateField(filed_name)
    filed_name = ogr.FieldDefn("type", ogr.OFTString)
    layer.CreateField(filed_name)
    defn = layer.GetLayerDefn()

    with open(input_pkl, 'rb') as f:
        polygons = pickle.load(f)

    for polygon in polygons:
        feature = ogr.Feature(defn)
        feature.SetField('type', polygon[0][0])
        feature.SetField("Confidence", polygon[0][1])
        multipolygon = ogr.Geometry(ogr.wkbMultiPolygon)
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in polygon[1:]:
            new_point = transformer(geoTransform, point[0], point[1])
            if not ds_image:
                new_point = ct.TransformPoint(new_point[0], new_point[1])
            ring.AddPoint(new_point[0], new_point[1])

        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        multipolygon.AddGeometry(poly)
        wkt = multipolygon.ExportToWkt()

        rectangle = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(rectangle)
        layer.CreateFeature(feature)
        feature.Destroy()
    datasource.Destroy()
    if finsh_flag:
        finsh_flag.value += 1


if __name__ == '__main__':
    pass
