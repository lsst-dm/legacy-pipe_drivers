import argparse

from lsst.pipe.base import Struct, TaskRunner
from lsst.pipe.tasks.coaddBase import CoaddDataIdContainer
from lsst.pipe.tasks.selectImages import BaseSelectImagesTask, BaseExposureInfo


class ButlerTaskRunner(TaskRunner):
    """Get a butler into the Task scripts"""
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        """Task.runDataRef should receive a butler in the kwargs"""
        return TaskRunner.getTargetList(parsedCmd, butler=parsedCmd.butler, **kwargs)


def getDataRef(butler, dataId, datasetType="raw"):
    """Construct a dataRef from a butler and data identifier"""
    dataRefList = [ref for ref in butler.subset(datasetType, **dataId)]
    assert len(dataRefList) == 1
    return dataRefList[0]


class NullSelectImagesTask(BaseSelectImagesTask):
    """Select images by taking everything we're given without further examination

    This is useful if the examination (e.g., Wcs checking) has been performed
    previously, and we've been provided a good list.
    """

    def runDataRef(self, patchRef, coordList, makeDataRefList=True, selectDataList=[]):
        return Struct(
            dataRefList=[s.dataRef for s in selectDataList],
            exposureInfoList=[BaseExposureInfo(
                s.dataRef.dataId, None) for s in selectDataList],
        )


class TractDataIdContainer(CoaddDataIdContainer):

    def makeDataRefList(self, namespace):
        """Make self.refList from self.idList

        It's difficult to make a data reference that merely points to an entire
        tract: there is no data product solely at the tract level.  Instead, we
        generate a list of data references for patches within the tract.
        """
        datasetType = namespace.config.coaddName + "Coadd_calexp"
        validKeys = set(["tract", "filter", "patch"])

        def getPatchRefList(tract):
            return [namespace.butler.dataRef(datasetType=datasetType,
                                             tract=tract.getId(),
                                             filter=dataId["filter"],
                                             patch="%d,%d" % patch.getIndex())
                    for patch in tract]

        tractRefs = {}  # Data references for each tract
        for dataId in self.idList:
            for key in validKeys:
                if key in ("tract", "patch",):
                    # Will deal with these explicitly
                    continue
                if key not in dataId:
                    raise argparse.ArgumentError(
                        None, "--id must include " + key)

            skymap = self.getSkymap(namespace)

            if "tract" in dataId:
                tractId = dataId["tract"]
                if tractId not in tractRefs:
                    tractRefs[tractId] = []
                if "patch" in dataId:
                    tractRefs[tractId].append(namespace.butler.dataRef(datasetType=datasetType, tract=tractId,
                                                                       filter=dataId[
                                                                           'filter'],
                                                                       patch=dataId['patch']))
                else:
                    tractRefs[tractId] += getPatchRefList(skymap[tractId])
            else:
                tractRefs = dict((tract.getId(), tractRefs.get(tract.getId(), []) +
                                  getPatchRefList(tract)) for tract in skymap)

        self.refList = list(tractRefs.values())
