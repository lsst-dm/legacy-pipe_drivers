from lsst.pipe.base import Struct
from lsst.pipe.tasks.selectImages import BaseSelectImagesTask, BaseExposureInfo

def getDataRef(butler, dataId, datasetType="raw"):
    """Construct a dataRef from a butler and data identifier"""
    return butler.dataRef(datasetType, **dataId)

class NullSelectImagesTask(BaseSelectImagesTask):
    """Select images by taking everything we're given without further examination
    This is useful if the examination (e.g., Wcs checking) has been performed
    previously, and we've been provided a good list.
    """
    def runDataRef(self, patchRef, coordList, makeDataRefList=True, selectDataList=[]):
        return Struct(
            dataRefList = [s.dataRef for s in selectDataList],
            exposureInfoList = [BaseExposureInfo(s.dataRef.dataId, None) for s in selectDataList],
            )   
