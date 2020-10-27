from .coco import CocoDataset
from .builder import DATASETS

@DATASETS.register_module()
class BuildingDataset(CocoDataset):
    CLASSES = ('building',)