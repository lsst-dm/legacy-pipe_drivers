from __future__ import absolute_import, division, print_function

from builtins import zip
from builtins import map

import lsst.afw.image as afwImage
import lsst.afw.geom as afwGeom
from lsst.afw.fits.fitsLib import FitsError
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError, NODE
from lsst.geom import convexHull
from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import Struct, ArgumentParser
from lsst.pipe.tasks.coaddBase import CoaddTaskRunner
from lsst.pipe.tasks.makeCoaddTempExp import MakeCoaddTempExpTask
from lsst.pipe.tasks.multiBand import DetectCoaddSourcesTask
from lsst.pipe.tasks.selectImages import WcsSelectImagesTask
from lsst.pipe.tasks.assembleCoadd import SafeClipAssembleCoaddTask
from lsst.pipe.drivers.utils import getDataRef, NullSelectImagesTask, TractDataIdContainer


class CoaddDriverConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name for coadd")
    select = ConfigurableField(
        target=WcsSelectImagesTask, doc="Select images to process")
    makeCoaddTempExp = ConfigurableField(
        target=MakeCoaddTempExpTask, doc="Warp images to sky")
    doBackgroundReference = Field(
        dtype=bool, default=False, doc="Build background reference?")
    backgroundReference = ConfigurableField(
        target=NullSelectImagesTask, doc="Build background reference")
    assembleCoadd = ConfigurableField(
        target=SafeClipAssembleCoaddTask, doc="Assemble warps into coadd")
    detectCoaddSources = ConfigurableField(
        target=DetectCoaddSourcesTask, doc="Detect sources on coadd")
    doOverwriteCoadd = Field(dtype=bool, default=False, doc="Overwrite coadd?")

    def setDefaults(self):
        self.makeCoaddTempExp.select.retarget(NullSelectImagesTask)
        self.assembleCoadd.select.retarget(NullSelectImagesTask)
        self.makeCoaddTempExp.doOverwrite = False
        self.assembleCoadd.doWrite = False
        self.assembleCoadd.doMatchBackgrounds = False
        self.makeCoaddTempExp.bgSubtracted = True
        self.assembleCoadd.badMaskPlanes = [
            'BAD', 'EDGE', 'SAT', 'INTRP', 'NO_DATA']

    def validate(self):
        if self.makeCoaddTempExp.coaddName != self.coaddName:
            raise RuntimeError(
                "makeCoaddTempExp.coaddName and coaddName don't match")
        if self.assembleCoadd.coaddName != self.coaddName:
            raise RuntimeError(
                "assembleCoadd.coaddName and coaddName don't match")


class CoaddDriverTaskRunner(CoaddTaskRunner):

    @staticmethod
    def getTargetList(parsedCmd, **kwargs):
        """!Get bare butler into Task

        @param parsedCmd results of parsing command input
        """
        kwargs["butler"] = parsedCmd.butler
        kwargs["selectIdList"] = [
            ref.dataId for ref in parsedCmd.selectId.refList]
        return [(parsedCmd.id.refList, kwargs), ]


class CoaddDriverTask(BatchPoolTask):
    ConfigClass = CoaddDriverConfig
    _DefaultName = "coaddDriver"
    RunnerClass = CoaddDriverTaskRunner

    def __init__(self, *args, **kwargs):
        BatchPoolTask.__init__(self, *args, **kwargs)
        self.makeSubtask("select")
        self.makeSubtask("makeCoaddTempExp")
        self.makeSubtask("backgroundReference")
        self.makeSubtask("assembleCoadd")
        self.makeSubtask("detectCoaddSources")

    @classmethod
    def _makeArgumentParser(cls, **kwargs):
        """!Build argument parser

        Selection references are not cheap (reads Wcs), so are generated
        only if we're not doing a batch submission.
        """
        parser = ArgumentParser(name=cls._DefaultName)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=TractDataIdContainer)
        parser.add_id_argument(
            "--selectId", "calexp", help="data ID, e.g. --selectId visit=6789 ccd=0..9")
        return parser

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        """!
        Return walltime request for batch job

        @param time: Requested time per iteration
        @param parsedCmd: Results of argument parsing
        @param numCores: Number of cores
        @return float walltime request length
        """
        numTargets = len(parsedCmd.selectId.refList)
        return time*numTargets/float(numCores)

    @abortOnError
    def run(self, tractPatchRefList, butler, selectIdList=[]):
        """!Determine which tracts are non-empty before processing

        @param tractPatchRefList: List of tracts and patches to include in the coaddition
        @param butler: butler reference object
        @param selectIdList: List of data Ids (i.e. visit, ccd) to consider when making the coadd
        @return list of references to sel.runTract function evaluation for each tractPatchRefList member
        """
        pool = Pool("tracts")
        pool.storeSet(butler=butler, skymap=butler.get(
            self.config.coaddName + "Coadd_skyMap"))
        tractIdList = []
        for patchRefList in tractPatchRefList:
            tractSet = set([patchRef.dataId["tract"]
                            for patchRef in patchRefList])
            assert len(tractSet) == 1
            tractIdList.append(tractSet.pop())

        selectDataList = [data for data in pool.mapNoBalance(self.readSelection, selectIdList) if
                          data is not None]
        nonEmptyList = pool.mapNoBalance(
            self.checkTract, tractIdList, selectDataList)
        tractPatchRefList = [patchRefList for patchRefList, nonEmpty in
                             zip(tractPatchRefList, nonEmptyList) if nonEmpty]
        self.log.info("Non-empty tracts (%d): %s" % (len(tractPatchRefList),
                                                     [patchRefList[0].dataId["tract"] for patchRefList in
                                                      tractPatchRefList]))

        # Install the dataRef in the selectDataList
        for data in selectDataList:
            data.dataRef = getDataRef(butler, data.dataId, "calexp")

        # Process the non-empty tracts
        return [self.runTract(patchRefList, butler, selectDataList) for patchRefList in tractPatchRefList]

    @abortOnError
    def runTract(self, patchRefList, butler, selectDataList=[]):
        """!Run stacking on a tract

        This method only runs on the master node.

        @param patchRefList: List of patch data references for tract
        @param butler: Data butler
        @param selectDataList: List of SelectStruct for inputs
        """
        pool = Pool("stacker")
        pool.cacheClear()
        pool.storeSet(butler=butler, warpType=self.config.coaddName + "Coadd_directWarp",
                      coaddType=self.config.coaddName + "Coadd")
        patchIdList = [patchRef.dataId for patchRef in patchRefList]

        selectedData = pool.map(self.warp, patchIdList, selectDataList)
        if self.config.doBackgroundReference:
            self.backgroundReference.run(patchRefList, selectDataList)

        def refNamer(patchRef):
            return tuple(map(int, patchRef.dataId["patch"].split(",")))

        lookup = dict(zip(map(refNamer, patchRefList), selectedData))
        coaddData = [Struct(patchId=patchRef.dataId, selectDataList=lookup[refNamer(patchRef)]) for
                     patchRef in patchRefList]
        pool.map(self.coadd, coaddData)

    def readSelection(self, cache, selectId):
        """!Read Wcs of selected inputs

        This method only runs on slave nodes.
        This method is similar to SelectDataIdContainer.makeDataRefList,
        creating a Struct like a SelectStruct, except with a dataId instead
        of a dataRef (to ease MPI).

        @param cache: Pool cache
        @param selectId: Data identifier for selected input
        @return a SelectStruct with a dataId instead of dataRef
        """
        try:
            ref = getDataRef(cache.butler, selectId, "calexp")
            self.log.info("Reading Wcs from %s" % (selectId,))
            md = ref.get("calexp_md", immediate=True)
            wcs = afwImage.makeWcs(md)
            data = Struct(dataId=selectId, wcs=wcs, dims=(
                md.get("NAXIS1"), md.get("NAXIS2")))
        except FitsError:
            self.log.warn("Unable to construct Wcs from %s" % (selectId,))
            return None
        return data

    def checkTract(self, cache, tractId, selectIdList):
        """!Check whether a tract has any overlapping inputs

        This method only runs on slave nodes.

        @param cache: Pool cache
        @param tractId: Data identifier for tract
        @param selectDataList: List of selection data
        @return whether tract has any overlapping inputs
        """
        skymap = cache.skymap
        tract = skymap[tractId]
        tractWcs = tract.getWcs()
        tractPoly = convexHull([tractWcs.pixelToSky(afwGeom.Point2D(coord)).getVector() for
                                coord in tract.getBBox().getCorners()])

        for selectData in selectIdList:
            if not hasattr(selectData, "poly"):
                wcs = selectData.wcs
                dims = selectData.dims
                box = afwGeom.Box2D(afwGeom.Point2D(0, 0),
                                    afwGeom.Point2D(*dims))
                selectData.poly = convexHull([wcs.pixelToSky(coord).getVector()
                                              for coord in box.getCorners()])
            if tractPoly.intersects(selectData.poly):
                return True
        return False

    def warp(self, cache, patchId, selectDataList):
        """!Warp all images for a patch

        Only slave nodes execute this method.

        Because only one argument may be passed, it is expected to
        contain multiple elements, which are:

        @param patchRef: data reference for patch
        @param selectDataList: List of SelectStruct for inputs
        @return selectDataList with non-overlapping elements removed
        """
        patchRef = getDataRef(cache.butler, patchId, cache.coaddType)
        selectDataList = self.selectExposures(patchRef, selectDataList)
        with self.logOperation("warping %s" % (patchRef.dataId,), catch=True):
            self.makeCoaddTempExp.run(patchRef, selectDataList)
        return selectDataList

    def coadd(self, cache, data):
        """!Construct coadd for a patch and measure

        Only slave nodes execute this method.

        Because only one argument may be passed, it is expected to
        contain multiple elements, which are:

        @param patchRef: data reference for patch
        @param selectDataList: List of SelectStruct for inputs
        """
        patchRef = getDataRef(cache.butler, data.patchId, cache.coaddType)
        selectDataList = data.selectDataList
        coadd = None
        with self.logOperation("coadding %s" % (patchRef.dataId,), catch=True):
            if self.config.doOverwriteCoadd or not patchRef.datasetExists(cache.coaddType):
                coaddResults = self.assembleCoadd.run(patchRef, selectDataList)
                if coaddResults is not None:
                    coadd = coaddResults.coaddExposure
            elif patchRef.datasetExists(cache.coaddType):
                self.log.info("%s: Reading coadd %s" % (NODE, patchRef.dataId))
                coadd = patchRef.get(cache.coaddType, immediate=True)

        if coadd is None:
            return

        with self.logOperation("detection on %s" % (patchRef.dataId,), catch=True):
            idFactory = self.detectCoaddSources.makeIdFactory(patchRef)
            # This includes background subtraction, so do it before writing the
            # coadd
            detResults = self.detectCoaddSources.runDetection(coadd, idFactory)
            self.detectCoaddSources.write(coadd, detResults, patchRef)

    def selectExposures(self, patchRef, selectDataList):
        """!Select exposures to operate upon, via the SelectImagesTask

        This is very similar to CoaddBaseTask.selectExposures, except we return
        a list of SelectStruct (same as the input), so we can plug the results into
        future uses of SelectImagesTask.

        @param patchRef data reference to a particular patch
        @param selectDataList list of references to specific data products (i.e. visit, ccd)
        @return filtered list of SelectStruct
        """
        def key(dataRef):
            return tuple(dataRef.dataId[k] for k in sorted(dataRef.dataId))
        inputs = dict((key(select.dataRef), select)
                      for select in selectDataList)
        skyMap = patchRef.get(self.config.coaddName + "Coadd_skyMap")
        tract = skyMap[patchRef.dataId["tract"]]
        patch = tract[(tuple(int(i)
                             for i in patchRef.dataId["patch"].split(",")))]
        bbox = patch.getOuterBBox()
        wcs = tract.getWcs()
        cornerPosList = afwGeom.Box2D(bbox).getCorners()
        coordList = [wcs.pixelToSky(pos) for pos in cornerPosList]
        dataRefList = self.select.runDataRef(
            patchRef, coordList, selectDataList=selectDataList).dataRefList
        return [inputs[key(dataRef)] for dataRef in dataRefList]

    def writeMetadata(self, dataRef):
        pass
