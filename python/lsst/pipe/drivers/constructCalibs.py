import sys
import math
import time
import argparse
import traceback
import collections

import numpy as np

from astro_metadata_translator import merge_headers, ObservationGroup
from astro_metadata_translator.serialize import dates_to_fits

from lsst.pex.config import Config, ConfigurableField, Field, ListField, ConfigField
from lsst.pipe.base import Task, Struct, TaskRunner, ArgumentParser
import lsst.daf.base as dafBase
import lsst.afw.math as afwMath
import lsst.afw.detection as afwDet
import lsst.afw.image as afwImage
import lsst.geom as geom
import lsst.meas.algorithms as measAlg
from lsst.pipe.tasks.repair import RepairTask
from lsst.ip.isr import IsrTask

from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, NODE
from lsst.pipe.drivers.background import (SkyMeasurementTask, FocalPlaneBackground,
                                          FocalPlaneBackgroundConfig, MaskObjectsTask)
from lsst.pipe.drivers.visualizeVisit import makeCameraImage

from .checksum import checksum
from .utils import getDataRef


class CalibStatsConfig(Config):
    """Parameters controlling the measurement of background statistics"""
    stat = Field(doc="Statistic to use to estimate background (from lsst.afw.math)", dtype=int,
                 default=int(afwMath.MEANCLIP))
    clip = Field(doc="Clipping threshold for background",
                 dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for background",
                  dtype=int, default=3)
    maxVisitsToCalcErrorFromInputVariance = Field(
        doc="Maximum number of visits to estimate variance from input variance, not per-pixel spread",
        dtype=int, default=2)
    mask = ListField(doc="Mask planes to reject",
                     dtype=str, default=["DETECTED", "BAD", "NO_DATA"])


class CalibStatsTask(Task):
    """Measure statistics on the background

    This can be useful for scaling the background, e.g., for flats and fringe frames.
    """
    ConfigClass = CalibStatsConfig

    def run(self, exposureOrImage):
        """!Measure a particular statistic on an image (of some sort).

        @param exposureOrImage    Exposure, MaskedImage or Image.
        @return Value of desired statistic
        """
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        try:
            image = exposureOrImage.getMaskedImage()
        except Exception:
            try:
                image = exposureOrImage.getImage()
            except Exception:
                image = exposureOrImage

        return afwMath.makeStatistics(image, self.config.stat, stats).getValue()


class CalibCombineConfig(Config):
    """Configuration for combining calib images"""
    rows = Field(doc="Number of rows to read at a time",
                 dtype=int, default=512)
    mask = ListField(doc="Mask planes to respect", dtype=str,
                     default=["SAT", "DETECTED", "INTRP"])
    combine = Field(doc="Statistic to use for combination (from lsst.afw.math)", dtype=int,
                    default=int(afwMath.MEANCLIP))
    clip = Field(doc="Clipping threshold for combination",
                 dtype=float, default=3.0)
    nIter = Field(doc="Clipping iterations for combination",
                  dtype=int, default=3)
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")


class CalibCombineTask(Task):
    """Task to combine calib images"""
    ConfigClass = CalibCombineConfig

    def __init__(self, *args, **kwargs):
        Task.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")

    def run(self, sensorRefList, expScales=None, finalScale=None, inputName="postISRCCD"):
        """!Combine calib images for a single sensor

        @param sensorRefList   List of data references to combine (for a single sensor)
        @param expScales       List of scales to apply for each exposure
        @param finalScale      Desired scale for final combined image
        @param inputName       Data set name for inputs
        @return combined image
        """
        width, height = self.getDimensions(sensorRefList)
        stats = afwMath.StatisticsControl(self.config.clip, self.config.nIter,
                                          afwImage.Mask.getPlaneBitMask(self.config.mask))
        numImages = len(sensorRefList)
        if numImages < 1:
            raise RuntimeError("No valid input data")
        if numImages < self.config.stats.maxVisitsToCalcErrorFromInputVariance:
            stats.setCalcErrorFromInputVariance(True)

        # Combine images
        combined = afwImage.MaskedImageF(width, height)
        imageList = [None]*numImages
        for start in range(0, height, self.config.rows):
            rows = min(self.config.rows, height - start)
            box = geom.Box2I(geom.Point2I(0, start),
                             geom.Extent2I(width, rows))
            subCombined = combined.Factory(combined, box)

            for i, sensorRef in enumerate(sensorRefList):
                if sensorRef is None:
                    imageList[i] = None
                    continue
                exposure = sensorRef.get(inputName + "_sub", bbox=box)
                if expScales is not None:
                    self.applyScale(exposure, expScales[i])
                imageList[i] = exposure.getMaskedImage()

            self.combine(subCombined, imageList, stats)

        if finalScale is not None:
            background = self.stats.run(combined)
            self.log.info("%s: Measured background of stack is %f; adjusting to %f" %
                          (NODE, background, finalScale))
            combined *= finalScale / background

        return combined

    def getDimensions(self, sensorRefList, inputName="postISRCCD"):
        """Get dimensions of the inputs"""
        dimList = []
        for sensorRef in sensorRefList:
            if sensorRef is None:
                continue
            md = sensorRef.get(inputName + "_md")
            dimList.append(afwImage.bboxFromMetadata(md).getDimensions())
        return getSize(dimList)

    def applyScale(self, exposure, scale=None):
        """Apply scale to input exposure

        This implementation applies a flux scaling: the input exposure is
        divided by the provided scale.
        """
        if scale is not None:
            mi = exposure.getMaskedImage()
            mi /= scale

    def combine(self, target, imageList, stats):
        """!Combine multiple images

        @param target      Target image to receive the combined pixels
        @param imageList   List of input images
        @param stats       Statistics control
        """
        images = [img for img in imageList if img is not None]
        afwMath.statisticsStack(target, images, afwMath.Property(self.config.combine), stats)


def getSize(dimList):
    """Determine a consistent size, given a list of image sizes"""
    dim = set((w, h) for w, h in dimList)
    dim.discard(None)
    if len(dim) != 1:
        raise RuntimeError("Inconsistent dimensions: %s" % dim)
    return dim.pop()


def dictToTuple(dict_, keys):
    """!Return a tuple of specific values from a dict

    This provides a hashable representation of the dict from certain keywords.
    This can be useful for creating e.g., a tuple of the values in the DataId
    that identify the CCD.

    @param dict_  dict to parse
    @param keys  keys to extract (order is important)
    @return tuple of values
    """
    return tuple(dict_[k] for k in keys)


def getCcdIdListFromExposures(expRefList, level="sensor", ccdKeys=["ccd"]):
    """!Determine a list of CCDs from exposure references

    This essentially inverts the exposure-level references (which
    provides a list of CCDs for each exposure), by providing
    a dataId list for each CCD.  Consider an input list of exposures
    [e1, e2, e3], and each exposure has CCDs c1 and c2.  Then this
    function returns:

        {(c1,): [e1c1, e2c1, e3c1], (c2,): [e1c2, e2c2, e3c2]}

    This is a dict whose keys are tuples of the identifying values of a
    CCD (usually just the CCD number) and the values are lists of dataIds
    for that CCD in each exposure.  A missing dataId is given the value
    None.

    @param expRefList   List of data references for exposures
    @param level        Level for the butler to generate CCDs
    @param ccdKeys      DataId keywords that identify a CCD
    @return dict of data identifier lists for each CCD;
            keys are values of ccdKeys in order
    """
    expIdList = [[ccdRef.dataId for ccdRef in expRef.subItems(
        level)] for expRef in expRefList]

    # Determine what additional keys make a CCD from an exposure
    if len(ccdKeys) != len(set(ccdKeys)):
        raise RuntimeError("Duplicate keys found in ccdKeys: %s" % ccdKeys)
    ccdNames = set()  # Set of tuples which are values for each of the CCDs in an exposure
    for ccdIdList in expIdList:
        for ccdId in ccdIdList:
            name = dictToTuple(ccdId, ccdKeys)
            ccdNames.add(name)

    # Turn the list of CCDs for each exposure into a list of exposures for
    # each CCD
    ccdLists = {}
    for n, ccdIdList in enumerate(expIdList):
        for ccdId in ccdIdList:
            name = dictToTuple(ccdId, ccdKeys)
            if name not in ccdLists:
                ccdLists[name] = []
            ccdLists[name].append(ccdId)

    for ccd in ccdLists:
        # Sort the list by the dataId values (ordered by key)
        ccdLists[ccd] = sorted(ccdLists[ccd], key=lambda dd: dictToTuple(dd, sorted(dd.keys())))

    return ccdLists


def mapToMatrix(pool, func, ccdIdLists, *args, **kwargs):
    """Generate a matrix of results using pool.map

    The function should have the call signature:
        func(cache, dataId, *args, **kwargs)

    We return a dict mapping 'ccd name' to a list of values for
    each exposure.

    @param pool  Process pool
    @param func  Function to call for each dataId
    @param ccdIdLists  Dict of data identifier lists for each CCD name
    @return matrix of results
    """
    dataIdList = sum(ccdIdLists.values(), [])
    resultList = pool.map(func, dataIdList, *args, **kwargs)
    # Piece everything back together
    data = dict((ccdName, [None] * len(expList)) for ccdName, expList in ccdIdLists.items())
    indices = dict(sum([[(tuple(dataId.values()) if dataId is not None else None, (ccdName, expNum))
                         for expNum, dataId in enumerate(expList)]
                        for ccdName, expList in ccdIdLists.items()], []))
    for dataId, result in zip(dataIdList, resultList):
        if dataId is None:
            continue
        ccdName, expNum = indices[tuple(dataId.values())]
        data[ccdName][expNum] = result
    return data


class CalibIdAction(argparse.Action):
    """Split name=value pairs and put the result in a dict"""

    def __call__(self, parser, namespace, values, option_string):
        output = getattr(namespace, self.dest, {})
        for nameValue in values:
            name, sep, valueStr = nameValue.partition("=")
            if not valueStr:
                parser.error("%s value %s must be in form name=value" %
                             (option_string, nameValue))
            output[name] = valueStr
        setattr(namespace, self.dest, output)


class CalibArgumentParser(ArgumentParser):
    """ArgumentParser for calibration construction"""

    def __init__(self, calibName, *args, **kwargs):
        """Add a --calibId argument to the standard pipe_base argument parser"""
        ArgumentParser.__init__(self, *args, **kwargs)
        self.calibName = calibName
        self.add_id_argument("--id", datasetType="raw",
                             help="input identifiers, e.g., --id visit=123 ccd=4")
        self.add_argument("--calibId", nargs="*", action=CalibIdAction, default={},
                          help="identifiers for calib, e.g., --calibId version=1",
                          metavar="KEY=VALUE1[^VALUE2[^VALUE3...]")

    def parse_args(self, *args, **kwargs):
        """Parse arguments

        Checks that the "--calibId" provided works.
        """
        namespace = ArgumentParser.parse_args(self, *args, **kwargs)

        keys = namespace.butler.getKeys(self.calibName)
        parsed = {}
        for name, value in namespace.calibId.items():
            if name not in keys:
                self.error(
                    "%s is not a relevant calib identifier key (%s)" % (name, keys))
            parsed[name] = keys[name](value)
        namespace.calibId = parsed

        return namespace


class CalibConfig(Config):
    """Configuration for constructing calibs"""
    clobber = Field(dtype=bool, default=True,
                    doc="Clobber existing processed images?")
    isr = ConfigurableField(target=IsrTask, doc="ISR configuration")
    dateObs = Field(dtype=str, default="dateObs",
                    doc="Key for observation date in exposure registry")
    dateCalib = Field(dtype=str, default="calibDate",
                      doc="Key for calib date in calib registry")
    filter = Field(dtype=str, default="filter",
                   doc="Key for filter name in exposure/calib registries")
    combination = ConfigurableField(
        target=CalibCombineTask, doc="Calib combination configuration")
    ccdKeys = ListField(dtype=str, default=["ccd"],
                        doc="DataId keywords specifying a CCD")
    visitKeys = ListField(dtype=str, default=["visit"],
                          doc="DataId keywords specifying a visit")
    calibKeys = ListField(dtype=str, default=[],
                          doc="DataId keywords specifying a calibration")
    doCameraImage = Field(dtype=bool, default=True, doc="Create camera overview image?")
    binning = Field(dtype=int, default=64, doc="Binning to apply for camera image")

    def setDefaults(self):
        self.isr.doWrite = False


class CalibTaskRunner(TaskRunner):
    """Get parsed values into the CalibTask.run"""
    @classmethod
    def getTargetList(cls, parsedCmd, **kwargs):
        return [dict(expRefList=parsedCmd.id.refList, butler=parsedCmd.butler, calibId=parsedCmd.calibId)]

    def __call__(self, args):
        """Call the Task with the kwargs from getTargetList"""
        task = self.TaskClass(config=self.config, log=self.log)
        exitStatus = 0                  # exit status for the shell
        if self.doRaise:
            result = task.runDataRef(**args)
        else:
            try:
                result = task.runDataRef(**args)
            except Exception as e:
                # n.b. The shell exit value is the number of dataRefs returning
                # non-zero, so the actual value used here is lost
                exitStatus = 1

                task.log.fatal("Failed: %s" % e)
                traceback.print_exc(file=sys.stderr)

        if self.doReturnResults:
            return Struct(
                exitStatus=exitStatus,
                args=args,
                metadata=task.metadata,
                result=result,
            )
        else:
            return Struct(
                exitStatus=exitStatus,
            )


class CalibTask(BatchPoolTask):
    """!Base class for constructing calibs.

    This should be subclassed for each of the required calib types.
    The subclass should be sure to define the following class variables:
    * _DefaultName: default name of the task, used by CmdLineTask
    * calibName: name of the calibration data set in the butler
    The subclass may optionally set:
    * filterName: filter name to give the resultant calib
    """
    ConfigClass = CalibConfig
    RunnerClass = CalibTaskRunner
    filterName = None
    calibName = None
    exposureTime = 1.0                  # sets this exposureTime in the output

    def __init__(self, *args, **kwargs):
        """Constructor"""
        BatchPoolTask.__init__(self, *args, **kwargs)
        self.makeSubtask("isr")
        self.makeSubtask("combination")

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        numCcds = len(parsedCmd.butler.get("camera"))
        numExps = len(cls.getTargetList(parsedCmd)[0]['expRefList'])
        numCycles = int(numCcds/float(numCores) + 0.5)
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        return CalibArgumentParser(calibName=cls.calibName, name=cls._DefaultName, *args, **kwargs)

    def runDataRef(self, expRefList, butler, calibId):
        """!Construct a calib from a list of exposure references

        This is the entry point, called by the TaskRunner.__call__

        Only the master node executes this method.

        @param expRefList  List of data references at the exposure level
        @param butler      Data butler
        @param calibId   Identifier dict for calib
        """
        if len(expRefList) < 1:
            raise RuntimeError("No valid input data")

        for expRef in expRefList:
            self.addMissingKeys(expRef.dataId, butler, self.config.ccdKeys, 'raw')

        outputId = self.getOutputId(expRefList, calibId)
        ccdIdLists = getCcdIdListFromExposures(
            expRefList, level="sensor", ccdKeys=self.config.ccdKeys)
        self.checkCcdIdLists(ccdIdLists)

        # Ensure we can generate filenames for each output
        outputIdItemList = list(outputId.items())
        for ccdName in ccdIdLists:
            dataId = dict([(k, ccdName[i]) for i, k in enumerate(self.config.ccdKeys)])
            dataId.update(outputIdItemList)
            self.addMissingKeys(dataId, butler)
            dataId.update(outputIdItemList)

            try:
                butler.get(self.calibName + "_filename", dataId)
            except Exception as e:
                raise RuntimeError(
                    "Unable to determine output filename \"%s_filename\" from %s: %s" %
                    (self.calibName, dataId, e))

        processPool = Pool("process")
        processPool.storeSet(butler=butler)

        # Scatter: process CCDs independently
        data = self.scatterProcess(processPool, ccdIdLists)

        # Gather: determine scalings
        scales = self.scale(ccdIdLists, data)

        combinePool = Pool("combine")
        combinePool.storeSet(butler=butler)

        # Scatter: combine
        calibs = self.scatterCombine(combinePool, outputId, ccdIdLists, scales)

        if self.config.doCameraImage:
            camera = butler.get("camera")
            # Convert indexing of calibs from "ccdName" to detector ID (as used by makeImageFromCamera)
            calibs = {butler.get("postISRCCD_detector",
                                 dict(zip(self.config.ccdKeys, ccdName))).getId(): calibs[ccdName]
                      for ccdName in ccdIdLists}

            try:
                cameraImage = self.makeCameraImage(camera, outputId, calibs)
                butler.put(cameraImage, self.calibName + "_camera", dataId)
            except Exception as exc:
                self.log.warn("Unable to create camera image: %s" % (exc,))

        return Struct(
            outputId=outputId,
            ccdIdLists=ccdIdLists,
            scales=scales,
            calibs=calibs,
            processPool=processPool,
            combinePool=combinePool,
        )

    def getOutputId(self, expRefList, calibId):
        """!Generate the data identifier for the output calib

        The mean date and the common filter are included, using keywords
        from the configuration.  The CCD-specific part is not included
        in the data identifier.

        @param expRefList  List of data references at exposure level
        @param calibId  Data identifier elements for the calib provided by the user
        @return data identifier
        """
        midTime = 0
        filterName = None
        for expRef in expRefList:
            butler = expRef.getButler()
            dataId = expRef.dataId

            midTime += self.getMjd(butler, dataId)
            thisFilter = self.getFilter(
                butler, dataId) if self.filterName is None else self.filterName
            if filterName is None:
                filterName = thisFilter
            elif filterName != thisFilter:
                raise RuntimeError("Filter mismatch for %s: %s vs %s" % (
                    dataId, thisFilter, filterName))

        midTime /= len(expRefList)
        date = str(dafBase.DateTime(
            midTime, dafBase.DateTime.MJD).toPython().date())

        outputId = {self.config.filter: filterName,
                    self.config.dateCalib: date}
        outputId.update(calibId)
        return outputId

    def getMjd(self, butler, dataId, timescale=dafBase.DateTime.UTC):
        """Determine the Modified Julian Date (MJD; in TAI) from a data identifier"""
        if self.config.dateObs in dataId:
            dateObs = dataId[self.config.dateObs]
        else:
            dateObs = butler.queryMetadata('raw', [self.config.dateObs], dataId)[0]
        if "T" not in dateObs:
            dateObs = dateObs + "T12:00:00.0Z"
        elif not dateObs.endswith("Z"):
            dateObs += "Z"

        return dafBase.DateTime(dateObs, timescale).get(dafBase.DateTime.MJD)

    def getFilter(self, butler, dataId):
        """Determine the filter from a data identifier"""
        filt = butler.queryMetadata('raw', [self.config.filter], dataId)[0]
        return filt

    def addMissingKeys(self, dataId, butler, missingKeys=None, calibName=None):
        if calibName is None:
            calibName = self.calibName

        if missingKeys is None:
            missingKeys = set(butler.getKeys(calibName).keys()) - set(dataId.keys())

        for k in missingKeys:
            try:
                v = butler.queryMetadata('raw', [k], dataId)  # n.b. --id refers to 'raw'
            except Exception:
                continue

            if len(v) == 0:         # failed to lookup value
                continue

            if len(v) == 1:
                dataId[k] = v[0]
            else:
                raise RuntimeError("No unique lookup for %s: %s" % (k, v))

    def updateMetadata(self, calibImage, exposureTime, darkTime=None, **kwargs):
        """!Update the metadata from the VisitInfo

        @param calibImage       The image whose metadata is to be set
        @param exposureTime     The exposure time for the image
        @param darkTime         The time since the last read (default: exposureTime)
        """

        if darkTime is None:
            darkTime = exposureTime     # avoid warning messages when using calibration products

        visitInfo = afwImage.VisitInfo(exposureTime=exposureTime, darkTime=darkTime, **kwargs)
        md = calibImage.getMetadata()

        afwImage.setVisitInfoMetadata(md, visitInfo)

    def scatterProcess(self, pool, ccdIdLists):
        """!Scatter the processing among the nodes

        We scatter each CCD independently (exposures aren't grouped together),
        to make full use of all available processors. This necessitates piecing
        everything back together in the same format as ccdIdLists afterwards.

        Only the master node executes this method.

        @param pool  Process pool
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Dict of lists of returned data for each CCD name
        """
        self.log.info("Scatter processing")
        return mapToMatrix(pool, self.process, ccdIdLists)

    def process(self, cache, ccdId, outputName="postISRCCD", **kwargs):
        """!Process a CCD, specified by a data identifier

        After processing, optionally returns a result (produced by
        the 'processResult' method) calculated from the processed
        exposure.  These results will be gathered by the master node,
        and is a means for coordinated scaling of all CCDs for flats,
        etc.

        Only slave nodes execute this method.

        @param cache  Process pool cache
        @param ccdId  Data identifier for CCD
        @param outputName  Output dataset name for butler
        @return result from 'processResult'
        """
        if ccdId is None:
            self.log.warn("Null identifier received on %s" % NODE)
            return None
        sensorRef = getDataRef(cache.butler, ccdId)
        if self.config.clobber or not sensorRef.datasetExists(outputName):
            self.log.info("Processing %s on %s" % (ccdId, NODE))
            try:
                exposure = self.processSingle(sensorRef, **kwargs)
            except Exception as e:
                self.log.warn("Unable to process %s: %s" % (ccdId, e))
                raise
                return None
            self.processWrite(sensorRef, exposure)
        else:
            self.log.info(
                "Using previously persisted processed exposure for %s" % (sensorRef.dataId,))
            exposure = sensorRef.get(outputName)
        return self.processResult(exposure)

    def processSingle(self, dataRef):
        """Process a single CCD, specified by a data reference

        Generally, this simply means doing ISR.

        Only slave nodes execute this method.
        """
        return self.isr.runDataRef(dataRef).exposure

    def processWrite(self, dataRef, exposure, outputName="postISRCCD"):
        """!Write the processed CCD

        We need to write these out because we can't hold them all in
        memory at once.

        Only slave nodes execute this method.

        @param dataRef     Data reference
        @param exposure    CCD exposure to write
        @param outputName  Output dataset name for butler.
        """
        dataRef.put(exposure, outputName)

    def processResult(self, exposure):
        """Extract processing results from a processed exposure

        This method generates what is gathered by the master node.
        This can be a background measurement or similar for scaling
        flat-fields.  It must be picklable!

        Only slave nodes execute this method.
        """
        return None

    def scale(self, ccdIdLists, data):
        """!Determine scaling across CCDs and exposures

        This is necessary mainly for flats, so as to determine a
        consistent scaling across the entire focal plane.  This
        implementation is simply a placeholder.

        Only the master node executes this method.

        @param ccdIdLists  Dict of data identifier lists for each CCD tuple
        @param data        Dict of lists of returned data for each CCD tuple
        @return dict of Struct(ccdScale: scaling for CCD,
                               expScales: scaling for each exposure
                               ) for each CCD tuple
        """
        self.log.info("Scale on %s" % NODE)
        return dict((name, Struct(ccdScale=None, expScales=[None] * len(ccdIdLists[name])))
                    for name in ccdIdLists)

    def scatterCombine(self, pool, outputId, ccdIdLists, scales):
        """!Scatter the combination of exposures across multiple nodes

        In this case, we can only scatter across as many nodes as
        there are CCDs.

        Only the master node executes this method.

        @param pool  Process pool
        @param outputId  Output identifier (exposure part only)
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @param scales  Dict of structs with scales, for each CCD name
        @param dict of binned images
        """
        self.log.info("Scatter combination")
        data = [Struct(ccdName=ccdName, ccdIdList=ccdIdLists[ccdName], scales=scales[ccdName]) for
                ccdName in ccdIdLists]
        images = pool.map(self.combine, data, outputId)
        return dict(zip(ccdIdLists.keys(), images))

    def getFullyQualifiedOutputId(self, ccdName, butler, outputId):
        """Get fully-qualified output data identifier

        We may need to look up keys that aren't in the output dataId.

        @param ccdName  Name tuple for CCD
        @param butler  Data butler
        @param outputId  Data identifier for combined image (exposure part only)
        @return fully-qualified output dataId
        """
        fullOutputId = {k: ccdName[i] for i, k in enumerate(self.config.ccdKeys)}
        fullOutputId.update(outputId)
        self.addMissingKeys(fullOutputId, butler)
        fullOutputId.update(outputId)  # must be after the call to queryMetadata in 'addMissingKeys'
        return fullOutputId

    def combine(self, cache, struct, outputId):
        """!Combine multiple exposures of a particular CCD and write the output

        Only the slave nodes execute this method.

        @param cache  Process pool cache
        @param struct  Parameters for the combination, which has the following components:
            * ccdName     Name tuple for CCD
            * ccdIdList   List of data identifiers for combination
            * scales      Scales to apply (expScales are scalings for each exposure,
                               ccdScale is final scale for combined image)
        @param outputId    Data identifier for combined image (exposure part only)
        @return binned calib image
        """
        outputId = self.getFullyQualifiedOutputId(struct.ccdName, cache.butler, outputId)
        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]
        self.log.info("Combining %s on %s" % (outputId, NODE))
        calib = self.combination.run(dataRefList, expScales=struct.scales.expScales,
                                     finalScale=struct.scales.ccdScale)

        if not hasattr(calib, "getMetadata"):
            if hasattr(calib, "getVariance"):
                calib = afwImage.makeExposure(calib)
            else:
                calib = afwImage.DecoratedImageF(calib.getImage())  # n.b. hardwires "F" for the output type

        self.calculateOutputHeaderFromRaws(cache.butler, calib, struct.ccdIdList, outputId)

        self.updateMetadata(calib, self.exposureTime)

        self.recordCalibInputs(cache.butler, calib,
                               struct.ccdIdList, outputId)

        self.interpolateNans(calib)

        self.write(cache.butler, calib, outputId)

        return afwMath.binImage(calib.getImage(), self.config.binning)

    def calculateOutputHeaderFromRaws(self, butler, calib, dataIdList, outputId):
        """!Calculate the output header from the raw headers.

        This metadata will go into the output FITS header. It will include all
        headers that are identical in all inputs.

        @param butler  Data butler
        @param calib  Combined calib exposure.
        @param dataIdList  List of data identifiers for calibration inputs
        @param outputId  Data identifier for output
        """
        header = calib.getMetadata()

        rawmd = [butler.get("raw_md", dataId) for dataId in dataIdList if
                 dataId is not None]

        merged = merge_headers(rawmd, mode="drop")

        # Place merged set into the PropertyList if a value is not
        # present already
        # Comments are not present in the merged version so copy them across
        for k, v in merged.items():
            if k not in header:
                comment = rawmd[0].getComment(k) if k in rawmd[0] else None
                header.set(k, v, comment=comment)

        # Create an observation group so we can add some standard headers
        # independent of the form in the input files.
        # Use try block in case we are dealing with unexpected data headers
        try:
            group = ObservationGroup(rawmd, pedantic=False)
        except Exception:
            group = None

        comments = {"TIMESYS": "Time scale for all dates",
                    "DATE-OBS": "Start date of earliest input observation",
                    "MJD-OBS": "[d] Start MJD of earliest input observation",
                    "DATE-END": "End date of oldest input observation",
                    "MJD-END": "[d] End MJD of oldest input observation",
                    "MJD-AVG": "[d] MJD midpoint of all input observations",
                    "DATE-AVG": "Midpoint date of all input observations"}

        if group is not None:
            oldest, newest = group.extremes()
            dateCards = dates_to_fits(oldest.datetime_begin, newest.datetime_end)
        else:
            # Fall back to setting a DATE-OBS from the calibDate
            dateCards = {"DATE-OBS": "{}T00:00:00.00".format(outputId[self.config.dateCalib])}
            comments["DATE-OBS"] = "Date of start of day of calibration midpoint"

        for k, v in dateCards.items():
            header.set(k, v, comment=comments.get(k, None))

    def recordCalibInputs(self, butler, calib, dataIdList, outputId):
        """!Record metadata including the inputs and creation details

        This metadata will go into the FITS header.

        @param butler  Data butler
        @param calib  Combined calib exposure.
        @param dataIdList  List of data identifiers for calibration inputs
        @param outputId  Data identifier for output
        """
        header = calib.getMetadata()
        header.set("OBSTYPE", self.calibName)  # Used by ingestCalibs.py

        # date, time, host, and root
        now = time.localtime()
        header.set("CALIB_CREATION_DATE", time.strftime("%Y-%m-%d", now))
        header.set("CALIB_CREATION_TIME", time.strftime("%X %Z", now))

        # Inputs
        visits = [str(dictToTuple(dataId, self.config.visitKeys)) for dataId in dataIdList if
                  dataId is not None]
        for i, v in enumerate(sorted(set(visits))):
            header.set("CALIB_INPUT_%d" % (i,), v)

        header.set("CALIB_ID", " ".join("%s=%s" % (key, value)
                                        for key, value in outputId.items()))
        checksum(calib, header)

    def interpolateNans(self, image):
        """Interpolate over NANs in the combined image

        NANs can result from masked areas on the CCD.  We don't want them getting
        into our science images, so we replace them with the median of the image.
        """
        if hasattr(image, "getMaskedImage"):  # Deal with Exposure vs Image
            self.interpolateNans(image.getMaskedImage().getVariance())
            image = image.getMaskedImage().getImage()
        if hasattr(image, "getImage"):  # Deal with DecoratedImage or MaskedImage vs Image
            image = image.getImage()
        array = image.getArray()
        bad = np.isnan(array)
        array[bad] = np.median(array[np.logical_not(bad)])

    def write(self, butler, exposure, dataId):
        """!Write the final combined calib

        Only the slave nodes execute this method

        @param butler  Data butler
        @param exposure  CCD exposure to write
        @param dataId  Data identifier for output
        """
        self.log.info("Writing %s on %s" % (dataId, NODE))
        butler.put(exposure, self.calibName, dataId)

    def makeCameraImage(self, camera, dataId, calibs):
        """!Create and write an image of the entire camera

        This is useful for judging the quality or getting an overview of
        the features of the calib.

        @param camera  Camera object
        @param dataId  Data identifier for output
        @param calibs  Dict mapping CCD detector ID to calib image
        """
        return makeCameraImage(camera, calibs, self.config.binning)

    def checkCcdIdLists(self, ccdIdLists):
        """Check that the list of CCD dataIds is consistent

        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Number of exposures, number of CCDs
        """
        visitIdLists = collections.defaultdict(list)
        for ccdName in ccdIdLists:
            for dataId in ccdIdLists[ccdName]:
                visitName = dictToTuple(dataId, self.config.visitKeys)
                visitIdLists[visitName].append(dataId)

        numExps = set(len(expList) for expList in ccdIdLists.values())
        numCcds = set(len(ccdList) for ccdList in visitIdLists.values())

        if len(numExps) != 1 or len(numCcds) != 1:
            # Presumably a visit somewhere doesn't have the full complement available.
            # Dump the information so the user can figure it out.
            self.log.warn("Number of visits for each CCD: %s",
                          {ccdName: len(ccdIdLists[ccdName]) for ccdName in ccdIdLists})
            self.log.warn("Number of CCDs for each visit: %s",
                          {vv: len(visitIdLists[vv]) for vv in visitIdLists})
            raise RuntimeError("Inconsistent number of exposures/CCDs")

        return numExps.pop(), numCcds.pop()


class BiasConfig(CalibConfig):
    """Configuration for bias construction.

    No changes required compared to the base class, but
    subclassed for distinction.
    """
    pass


class BiasTask(CalibTask):
    """Bias construction"""
    ConfigClass = BiasConfig
    _DefaultName = "bias"
    calibName = "bias"
    filterName = "NONE"  # Sets this filter name in the output
    exposureTime = 0.0   # sets this exposureTime in the output

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for bias construction"""
        config.isr.doBias = False
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False


class DarkConfig(CalibConfig):
    """Configuration for dark construction"""
    doRepair = Field(dtype=bool, default=True, doc="Repair artifacts?")
    psfFwhm = Field(dtype=float, default=3.0, doc="Repair PSF FWHM (pixels)")
    psfSize = Field(dtype=int, default=21, doc="Repair PSF size (pixels)")
    crGrow = Field(dtype=int, default=2, doc="Grow radius for CR (pixels)")
    repair = ConfigurableField(
        target=RepairTask, doc="Task to repair artifacts")

    def setDefaults(self):
        CalibConfig.setDefaults(self)
        self.combination.mask.append("CR")


class DarkTask(CalibTask):
    """Dark construction

    The only major difference from the base class is a cosmic-ray
    identification stage, and dividing each image by the dark time
    to generate images of the dark rate.
    """
    ConfigClass = DarkConfig
    _DefaultName = "dark"
    calibName = "dark"
    filterName = "NONE"  # Sets this filter name in the output

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("repair")

    @classmethod
    def applyOverrides(cls, config):
        """Overrides to apply for dark construction"""
        config.isr.doDark = False
        config.isr.doFlat = False
        config.isr.doFringe = False

    def processSingle(self, sensorRef):
        """Process a single CCD

        Besides the regular ISR, also masks cosmic-rays and divides each
        processed image by the dark time to generate images of the dark rate.
        The dark time is provided by the 'getDarkTime' method.
        """
        exposure = CalibTask.processSingle(self, sensorRef)

        if self.config.doRepair:
            psf = measAlg.DoubleGaussianPsf(self.config.psfSize, self.config.psfSize,
                                            self.config.psfFwhm/(2*math.sqrt(2*math.log(2))))
            exposure.setPsf(psf)
            self.repair.run(exposure, keepCRs=False)
            if self.config.crGrow > 0:
                mask = exposure.getMaskedImage().getMask().clone()
                mask &= mask.getPlaneBitMask("CR")
                fpSet = afwDet.FootprintSet(
                    mask, afwDet.Threshold(0.5))
                fpSet = afwDet.FootprintSet(fpSet, self.config.crGrow, True)
                fpSet.setMask(exposure.getMaskedImage().getMask(), "CR")

        mi = exposure.getMaskedImage()
        mi /= self.getDarkTime(exposure)
        return exposure

    def getDarkTime(self, exposure):
        """Retrieve the dark time for an exposure"""
        darkTime = exposure.getInfo().getVisitInfo().getDarkTime()
        if not np.isfinite(darkTime):
            raise RuntimeError("Non-finite darkTime")
        return darkTime


class FlatConfig(CalibConfig):
    """Configuration for flat construction"""
    iterations = Field(dtype=int, default=10,
                       doc="Number of iterations for scale determination")
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")


class FlatTask(CalibTask):
    """Flat construction

    The principal change from the base class involves gathering the background
    values from each image and using them to determine the scalings for the final
    combination.
    """
    ConfigClass = FlatConfig
    _DefaultName = "flat"
    calibName = "flat"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides for flat construction"""
        config.isr.doFlat = False
        config.isr.doFringe = False

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("stats")

    def processResult(self, exposure):
        return self.stats.run(exposure)

    def scale(self, ccdIdLists, data):
        """Determine the scalings for the final combination

        We have a matrix B_ij = C_i E_j, where C_i is the relative scaling
        of one CCD to all the others in an exposure, and E_j is the scaling
        of the exposure.  We convert everything to logarithms so we can work
        with a linear system.  We determine the C_i and E_j from B_ij by iteration,
        under the additional constraint that the average CCD scale is unity.

        This algorithm comes from Eugene Magnier and Pan-STARRS.
        """
        assert len(ccdIdLists.values()) > 0, "No successful CCDs"
        lengths = set([len(expList) for expList in ccdIdLists.values()])
        assert len(lengths) == 1, "Number of successful exposures for each CCD differs"
        assert tuple(lengths)[0] > 0, "No successful exposures"
        # Format background measurements into a matrix
        indices = dict((name, i) for i, name in enumerate(ccdIdLists))
        bgMatrix = np.array([[0.0] * len(expList) for expList in ccdIdLists.values()])
        for name in ccdIdLists:
            i = indices[name]
            bgMatrix[i] = [d if d is not None else np.nan for d in data[name]]

        numpyPrint = np.get_printoptions()
        np.set_printoptions(threshold=np.inf)
        self.log.info("Input backgrounds: %s" % bgMatrix)

        # Flat-field scaling
        numCcds = len(ccdIdLists)
        numExps = bgMatrix.shape[1]
        # log(Background) for each exposure/component
        bgMatrix = np.log(bgMatrix)
        bgMatrix = np.ma.masked_array(bgMatrix, ~np.isfinite(bgMatrix))
        # Initial guess at log(scale) for each component
        compScales = np.zeros(numCcds)
        expScales = np.array([(bgMatrix[:, i0] - compScales).mean() for i0 in range(numExps)])

        for iterate in range(self.config.iterations):
            compScales = np.array([(bgMatrix[i1, :] - expScales).mean() for i1 in range(numCcds)])
            bad = np.isnan(compScales)
            if np.any(bad):
                # Bad CCDs: just set them to the mean scale
                compScales[bad] = compScales[~bad].mean()
            expScales = np.array([(bgMatrix[:, i2] - compScales).mean() for i2 in range(numExps)])

            avgScale = np.average(np.exp(compScales))
            compScales -= np.log(avgScale)
            self.log.debug("Iteration %d exposure scales: %s", iterate, np.exp(expScales))
            self.log.debug("Iteration %d component scales: %s", iterate, np.exp(compScales))

        expScales = np.array([(bgMatrix[:, i3] - compScales).mean() for i3 in range(numExps)])

        if np.any(np.isnan(expScales)):
            raise RuntimeError("Bad exposure scales: %s --> %s" % (bgMatrix, expScales))

        expScales = np.exp(expScales)
        compScales = np.exp(compScales)

        self.log.info("Exposure scales: %s" % expScales)
        self.log.info("Component relative scaling: %s" % compScales)
        np.set_printoptions(**numpyPrint)

        return dict((ccdName, Struct(ccdScale=compScales[indices[ccdName]], expScales=expScales))
                    for ccdName in ccdIdLists)


class FringeConfig(CalibConfig):
    """Configuration for fringe construction"""
    stats = ConfigurableField(target=CalibStatsTask,
                              doc="Background statistics configuration")
    subtractBackground = ConfigurableField(target=measAlg.SubtractBackgroundTask,
                                           doc="Background configuration")
    detection = ConfigurableField(
        target=measAlg.SourceDetectionTask, doc="Detection configuration")
    detectSigma = Field(dtype=float, default=1.0,
                        doc="Detection PSF gaussian sigma")

    def setDefaults(self):
        CalibConfig.setDefaults(self)
        self.detection.reEstimateBackground = False


class FringeTask(CalibTask):
    """Fringe construction task

    The principal change from the base class is that the images are
    background-subtracted and rescaled by the background.

    XXX This is probably not right for a straight-up combination, as we
    are currently doing, since the fringe amplitudes need not scale with
    the continuum.

    XXX Would like to have this do PCA and generate multiple images, but
    that will take a bit of work with the persistence code.
    """
    ConfigClass = FringeConfig
    _DefaultName = "fringe"
    calibName = "fringe"

    @classmethod
    def applyOverrides(cls, config):
        """Overrides for fringe construction"""
        config.isr.doFringe = False

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("detection")
        self.makeSubtask("stats")
        self.makeSubtask("subtractBackground")

    def processSingle(self, sensorRef):
        """Subtract the background and normalise by the background level"""
        exposure = CalibTask.processSingle(self, sensorRef)
        bgLevel = self.stats.run(exposure)
        self.subtractBackground.run(exposure)
        mi = exposure.getMaskedImage()
        mi /= bgLevel
        footprintSets = self.detection.detectFootprints(
            exposure, sigma=self.config.detectSigma)
        mask = exposure.getMaskedImage().getMask()
        detected = 1 << mask.addMaskPlane("DETECTED")
        for fpSet in (footprintSets.positive, footprintSets.negative):
            if fpSet is not None:
                afwDet.setMaskFromFootprintList(
                    mask, fpSet.getFootprints(), detected)
        return exposure


class SkyConfig(CalibConfig):
    """Configuration for sky frame construction"""
    detectSigma = Field(dtype=float, default=2.0, doc="Detection PSF gaussian sigma")
    maskObjects = ConfigurableField(target=MaskObjectsTask,
                                    doc="Configuration for masking objects aggressively")
    largeScaleBackground = ConfigField(dtype=FocalPlaneBackgroundConfig,
                                       doc="Large-scale background configuration")
    sky = ConfigurableField(target=SkyMeasurementTask, doc="Sky measurement")
    maskThresh = Field(dtype=float, default=3.0, doc="k-sigma threshold for masking pixels")
    mask = ListField(dtype=str, default=["BAD", "SAT", "DETECTED", "NO_DATA"],
                     doc="Mask planes to consider as contaminated")


class SkyTask(CalibTask):
    """Task for sky frame construction

    The sky frame is a (relatively) small-scale background
    model, the response of the camera to the sky.

    To construct, we first remove a large-scale background (e.g., caused
    by moonlight) which may vary from image to image. Then we construct a
    model of the sky, which is essentially a binned version of the image
    (important configuration parameters: sky.background.[xy]BinSize).
    It is these models which are coadded to yield the sky frame.
    """
    ConfigClass = SkyConfig
    _DefaultName = "sky"
    calibName = "sky"

    def __init__(self, *args, **kwargs):
        CalibTask.__init__(self, *args, **kwargs)
        self.makeSubtask("maskObjects")
        self.makeSubtask("sky")

    def scatterProcess(self, pool, ccdIdLists):
        """!Scatter the processing among the nodes

        Only the master node executes this method, assigning work to the
        slaves.

        We measure and subtract off a large-scale background model across
        all CCDs, which requires a scatter/gather. Then we process the
        individual CCDs, subtracting the large-scale background model and
        the residual background model measured. These residuals will be
        combined for the sky frame.

        @param pool  Process pool
        @param ccdIdLists  Dict of data identifier lists for each CCD name
        @return Dict of lists of returned data for each CCD name
        """
        self.log.info("Scatter processing")

        numExps = set(len(expList) for expList in ccdIdLists.values()).pop()

        # First subtract off general gradients to make all the exposures look similar.
        # We want to preserve the common small-scale structure, which we will coadd.
        bgModelList = mapToMatrix(pool, self.measureBackground, ccdIdLists)

        backgrounds = {}
        scales = {}
        for exp in range(numExps):
            bgModels = [bgModelList[ccdName][exp] for ccdName in ccdIdLists]
            visit = set(tuple(ccdIdLists[ccdName][exp][key] for key in sorted(self.config.visitKeys)) for
                        ccdName in ccdIdLists)
            assert len(visit) == 1
            visit = visit.pop()
            bgModel = bgModels[0]
            for bg in bgModels[1:]:
                bgModel.merge(bg)
            self.log.info("Background model min/max for visit %s: %f %f", visit,
                          np.min(bgModel.getStatsImage().getArray()),
                          np.max(bgModel.getStatsImage().getArray()))
            backgrounds[visit] = bgModel
            scales[visit] = np.median(bgModel.getStatsImage().getArray())

        return mapToMatrix(pool, self.process, ccdIdLists, backgrounds=backgrounds, scales=scales)

    def measureBackground(self, cache, dataId):
        """!Measure background model for CCD

        This method is executed by the slaves.

        The background models for all CCDs in an exposure will be
        combined to form a full focal-plane background model.

        @param cache  Process pool cache
        @param dataId  Data identifier
        @return Bcakground model
        """
        dataRef = getDataRef(cache.butler, dataId)
        exposure = self.processSingleBackground(dataRef)

        # NAOJ prototype smoothed and then combined the entire image, but it shouldn't be any different
        # to bin and combine the binned images except that there's fewer pixels to worry about.
        config = self.config.largeScaleBackground
        camera = dataRef.get("camera")
        bgModel = FocalPlaneBackground.fromCamera(config, camera)
        bgModel.addCcd(exposure)
        return bgModel

    def processSingleBackground(self, dataRef):
        """!Process a single CCD for the background

        This method is executed by the slaves.

        Because we're interested in the background, we detect and mask astrophysical
        sources, and pixels above the noise level.

        @param dataRef  Data reference for CCD.
        @return processed exposure
        """
        if not self.config.clobber and dataRef.datasetExists("postISRCCD"):
            return dataRef.get("postISRCCD")
        exposure = CalibTask.processSingle(self, dataRef)

        self.maskObjects.run(exposure, self.config.mask)
        dataRef.put(exposure, "postISRCCD")
        return exposure

    def processSingle(self, dataRef, backgrounds, scales):
        """Process a single CCD, specified by a data reference

        We subtract the appropriate focal plane background model,
        divide by the appropriate scale and measure the background.

        Only slave nodes execute this method.

        @param dataRef  Data reference for single CCD
        @param backgrounds  Background model for each visit
        @param scales  Scales for each visit
        @return Processed exposure
        """
        visit = tuple(dataRef.dataId[key] for key in sorted(self.config.visitKeys))
        exposure = dataRef.get("postISRCCD", immediate=True)
        image = exposure.getMaskedImage()
        detector = exposure.getDetector()
        bbox = image.getBBox()

        bgModel = backgrounds[visit]
        bg = bgModel.toCcdBackground(detector, bbox)
        image -= bg.getImage()
        image /= scales[visit]

        bg = self.sky.measureBackground(exposure.getMaskedImage())
        dataRef.put(bg, "icExpBackground")
        return exposure

    def combine(self, cache, struct, outputId):
        """!Combine multiple background models of a particular CCD and write the output

        Only the slave nodes execute this method.

        @param cache  Process pool cache
        @param struct  Parameters for the combination, which has the following components:
            * ccdName     Name tuple for CCD
            * ccdIdList   List of data identifiers for combination
        @param outputId    Data identifier for combined image (exposure part only)
        @return binned calib image
        """
        outputId = self.getFullyQualifiedOutputId(struct.ccdName, cache.butler, outputId)
        dataRefList = [getDataRef(cache.butler, dataId) if dataId is not None else None for
                       dataId in struct.ccdIdList]
        self.log.info("Combining %s on %s" % (outputId, NODE))
        bgList = [dataRef.get("icExpBackground", immediate=True).clone() for dataRef in dataRefList]

        bgExp = self.sky.averageBackgrounds(bgList)

        self.recordCalibInputs(cache.butler, bgExp, struct.ccdIdList, outputId)
        cache.butler.put(bgExp, "sky", outputId)
        return afwMath.binImage(self.sky.exposureToBackground(bgExp).getImage(), self.config.binning)
