from __future__ import absolute_import, division, print_function
import os

from builtins import zip

from lsst.pex.config import Config, Field, ConfigurableField
from lsst.pipe.base import ArgumentParser, TaskRunner
from lsst.pipe.tasks.multiBand import (DetectCoaddSourcesTask,
                                       MergeDetectionsTask,
                                       DeblendCoaddSourcesTask,
                                       MeasureMergedCoaddSourcesTask,
                                       MergeMeasurementsTask,)
from lsst.ctrl.pool.parallel import BatchPoolTask
from lsst.ctrl.pool.pool import Pool, abortOnError
from lsst.meas.base.references import MultiBandReferencesTask
from lsst.meas.base.forcedPhotCoadd import ForcedPhotCoaddTask
from lsst.pipe.drivers.utils import getDataRef, TractDataIdContainer

import lsst.afw.table as afwTable


class MultiBandDriverConfig(Config):
    coaddName = Field(dtype=str, default="deep", doc="Name of coadd")
    doDetection = Field(dtype=bool, default=False,
                        doc="Re-run detection? (requires *Coadd dataset to have been written)")
    detectCoaddSources = ConfigurableField(target=DetectCoaddSourcesTask,
                                           doc="Detect sources on coadd")
    mergeCoaddDetections = ConfigurableField(
        target=MergeDetectionsTask, doc="Merge detections")
    deblendCoaddSources = ConfigurableField(target=DeblendCoaddSourcesTask, doc="Deblend merged detections")
    measureCoaddSources = ConfigurableField(target=MeasureMergedCoaddSourcesTask,
                                            doc="Measure merged and (optionally) deblended detections")
    mergeCoaddMeasurements = ConfigurableField(
        target=MergeMeasurementsTask, doc="Merge measurements")
    forcedPhotCoadd = ConfigurableField(target=ForcedPhotCoaddTask,
                                        doc="Forced measurement on coadded images")
    reprocessing = Field(
        dtype=bool, default=False,
        doc=("Are we reprocessing?\n\n"
             "This exists as a workaround for large deblender footprints causing large memory use "
             "and/or very slow processing.  We refuse to deblend those footprints when running on a cluster "
             "and return to reprocess on a machine with larger memory or more time "
             "if we consider those footprints important to recover."),
    )

    def setDefaults(self):
        Config.setDefaults(self)
        self.forcedPhotCoadd.references.retarget(MultiBandReferencesTask)

    def validate(self):
        for subtask in ("mergeCoaddDetections", "deblendCoaddSources", "measureCoaddSources",
                        "mergeCoaddMeasurements", "forcedPhotCoadd"):
            coaddName = getattr(self, subtask).coaddName
            if coaddName != self.coaddName:
                raise RuntimeError("%s.coaddName (%s) doesn't match root coaddName (%s)" %
                                   (subtask, coaddName, self.coaddName))


class MultiBandDriverTaskRunner(TaskRunner):
    """TaskRunner for running MultiBandTask

    This is similar to the lsst.pipe.base.ButlerInitializedTaskRunner,
    except that we have a list of data references instead of a single
    data reference being passed to the Task.run, and we pass the results
    of the '--reuse-outputs-from' command option to the Task constructor.
    """

    def __init__(self, TaskClass, parsedCmd, doReturnResults=False):
        TaskRunner.__init__(self, TaskClass, parsedCmd, doReturnResults)
        self.reuse = parsedCmd.reuse

    def makeTask(self, parsedCmd=None, args=None):
        """A variant of the base version that passes a butler argument to the task's constructor
        parsedCmd or args must be specified.
        """
        if parsedCmd is not None:
            butler = parsedCmd.butler
        elif args is not None:
            dataRefList, kwargs = args
            butler = dataRefList[0].butlerSubset.butler
        else:
            raise RuntimeError("parsedCmd or args must be specified")
        return self.TaskClass(config=self.config, log=self.log, butler=butler, reuse=self.reuse)


def unpickle(factory, args, kwargs):
    """Unpickle something by calling a factory"""
    return factory(*args, **kwargs)


class MultiBandDriverTask(BatchPoolTask):
    """Multi-node driver for multiband processing"""
    ConfigClass = MultiBandDriverConfig
    _DefaultName = "multiBandDriver"
    RunnerClass = MultiBandDriverTaskRunner

    def __init__(self, butler=None, schema=None, refObjLoader=None, reuse=tuple(), **kwargs):
        """!
        @param[in] butler: the butler can be used to retrieve schema or passed to the refObjLoader constructor
            in case it is needed.
        @param[in] schema: the schema of the source detection catalog used as input.
        @param[in] refObjLoader: an instance of LoadReferenceObjectsTasks that supplies an external reference
            catalog.  May be None if the butler argument is provided or all steps requiring a reference
            catalog are disabled.
        """
        BatchPoolTask.__init__(self, **kwargs)
        if schema is None:
            assert butler is not None, "Butler not provided"
            schema = butler.get(self.config.coaddName +
                                "Coadd_det_schema", immediate=True).schema
        self.butler = butler
        self.reuse = tuple(reuse)
        self.makeSubtask("detectCoaddSources")
        self.makeSubtask("mergeCoaddDetections", schema=schema)
        if self.config.measureCoaddSources.inputCatalog.startswith("deblended"):
            # Ensure that the output from deblendCoaddSources matches the input to measureCoaddSources
            self.measurementInput = self.config.measureCoaddSources.inputCatalog
            self.deblenderOutput = []
            if self.config.deblendCoaddSources.simultaneous:
                self.deblenderOutput.append("deblendedModel")
            else:
                self.deblenderOutput.append("deblendedFlux")
            if self.measurementInput not in self.deblenderOutput:
                err = "Measurement input '{0}' is not in the list of deblender output catalogs '{1}'"
                raise ValueError(err.format(self.measurementInput, self.deblenderOutput))

            self.makeSubtask("deblendCoaddSources",
                             schema=afwTable.Schema(self.mergeCoaddDetections.schema),
                             peakSchema=afwTable.Schema(self.mergeCoaddDetections.merged.getPeakSchema()),
                             butler=butler)
            measureInputSchema = afwTable.Schema(self.deblendCoaddSources.schema)
        else:
            measureInputSchema = afwTable.Schema(self.mergeCoaddDetections.schema)
        self.makeSubtask("measureCoaddSources", schema=measureInputSchema,
                         peakSchema=afwTable.Schema(
                             self.mergeCoaddDetections.merged.getPeakSchema()),
                         refObjLoader=refObjLoader, butler=butler)
        self.makeSubtask("mergeCoaddMeasurements", schema=afwTable.Schema(
            self.measureCoaddSources.schema))
        self.makeSubtask("forcedPhotCoadd", refSchema=afwTable.Schema(
            self.mergeCoaddMeasurements.schema))

    def __reduce__(self):
        """Pickler"""
        return unpickle, (self.__class__, [], dict(config=self.config, name=self._name,
                                                   parentTask=self._parentTask, log=self.log,
                                                   butler=self.butler, reuse=self.reuse))

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name=cls._DefaultName, *args, **kwargs)
        parser.add_id_argument("--id", "deepCoadd", help="data ID, e.g. --id tract=12345 patch=1,2",
                               ContainerClass=TractDataIdContainer)
        parser.addReuseOption(["detectCoaddSources", "mergeCoaddDetections", "measureCoaddSources",
                               "mergeCoaddMeasurements", "forcedPhotCoadd", "deblendCoaddSources"])
        return parser

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCpus):
        """!Return walltime request for batch job

        @param time: Requested time per iteration
        @param parsedCmd: Results of argument parsing
        @param numCores: Number of cores
        """
        numTargets = 0
        for refList in parsedCmd.id.refList:
            numTargets += len(refList)
        return time*numTargets/float(numCpus)

    @abortOnError
    def runDataRef(self, patchRefList):
        """!Run multiband processing on coadds

        Only the master node runs this method.

        No real MPI communication (scatter/gather) takes place: all I/O goes
        through the disk. We want the intermediate stages on disk, and the
        component Tasks are implemented around this, so we just follow suit.

        @param patchRefList:  Data references to run measurement
        """
        for patchRef in patchRefList:
            if patchRef:
                butler = patchRef.getButler()
                break
        else:
            raise RuntimeError("No valid patches")
        pool = Pool("all")
        pool.cacheClear()
        pool.storeSet(butler=butler)

        # MultiBand measurements require that the detection stage be completed
        # before measurements can be made.
        #
        # The configuration for coaddDriver.py allows detection to be turned
        # of in the event that fake objects are to be added during the
        # detection process.  This allows the long co-addition process to be
        # run once, and multiple different MultiBand reruns (with different
        # fake objects) to exist from the same base co-addition.
        #
        # However, we only re-run detection if doDetection is explicitly True
        # here (this should always be the opposite of coaddDriver.doDetection);
        # otherwise we have no way to tell reliably whether any detections
        # present in an input repo are safe to use.
        if self.config.doDetection:
            detectionList = []
            for patchRef in patchRefList:
                if ("detectCoaddSources" in self.reuse and
                        patchRef.datasetExists(self.config.coaddName + "Coadd_calexp", write=True)):
                    self.log.info("Skipping detectCoaddSources for %s; output already exists." %
                                  patchRef.dataId)
                    continue
                if not patchRef.datasetExists(self.config.coaddName + "Coadd"):
                    self.log.debug("Not processing %s; required input %sCoadd missing." %
                                   (patchRef.dataId, self.config.coaddName))
                    continue
                detectionList.append(patchRef)

            pool.map(self.runDetection, detectionList)

        patchRefList = [patchRef for patchRef in patchRefList if
                        patchRef.datasetExists(self.config.coaddName + "Coadd_calexp") and
                        patchRef.datasetExists(self.config.coaddName + "Coadd_det",
                                               write=self.config.doDetection)]
        dataIdList = [patchRef.dataId for patchRef in patchRefList]

        # Group by patch
        patches = {}
        tract = None
        for patchRef in patchRefList:
            dataId = patchRef.dataId
            if tract is None:
                tract = dataId["tract"]
            else:
                assert tract == dataId["tract"]

            patch = dataId["patch"]
            if patch not in patches:
                patches[patch] = []
            patches[patch].append(dataId)

        pool.map(self.runMergeDetections, patches.values())

        # Deblend merged detections, and test for reprocessing
        #
        # The reprocessing allows us to have multiple attempts at deblending large footprints. Large
        # footprints can suck up a lot of memory in the deblender, which means that when we process on a
        # cluster, we want to refuse to deblend them (they're flagged "deblend.parent-too-big"). But since
        # they may have astronomically interesting data, we want the ability to go back and reprocess them
        # with a more permissive configuration when we have more memory or processing time.
        #
        # self.runDeblendMerged will return whether there are any footprints in that image that required
        # reprocessing.  We need to convert that list of booleans into a dict mapping the patchId (x,y) to
        # a boolean. That tells us whether the merge measurement and forced photometry need to be re-run on
        # a particular patch.
        #
        # This determination of which patches need to be reprocessed exists only in memory (the measurements
        # have been written, clobbering the old ones), so if there was an exception we would lose this
        # information, leaving things in an inconsistent state (measurements, merged measurements and
        # forced photometry old). To attempt to preserve this status, we touch a file (dataset named
        # "deepCoadd_multibandReprocessing") --- if this file exists, we need to re-run the measurements,
        # merge and forced photometry.
        #
        # This is, hopefully, a temporary workaround until we can improve the
        # deblender.
        try:
            reprocessed = pool.map(self.runDeblendMerged, patches.values())
        finally:
            if self.config.reprocessing:
                patchReprocessing = {}
                for dataId, reprocess in zip(dataIdList, reprocessed):
                    patchId = dataId["patch"]
                    patchReprocessing[patchId] = patchReprocessing.get(
                        patchId, False) or reprocess
                # Persist the determination, to make error recover easier
                reprocessDataset = self.config.coaddName + "Coadd_multibandReprocessing"
                for patchId in patchReprocessing:
                    if not patchReprocessing[patchId]:
                        continue
                    dataId = dict(tract=tract, patch=patchId)
                    if patchReprocessing[patchId]:
                        filename = butler.get(
                            reprocessDataset + "_filename", dataId)[0]
                        open(filename, 'a').close()  # Touch file
                    elif butler.datasetExists(reprocessDataset, dataId):
                        # We must have failed at some point while reprocessing
                        # and we're starting over
                        patchReprocessing[patchId] = True

        # Only process patches that have been identified as needing it
        pool.map(self.runMeasurements, [dataId1 for dataId1 in dataIdList if not self.config.reprocessing or
                                        patchReprocessing[dataId1["patch"]]])
        pool.map(self.runMergeMeasurements, [idList for patchId, idList in patches.items() if
                                             not self.config.reprocessing or patchReprocessing[patchId]])
        pool.map(self.runForcedPhot, [dataId1 for dataId1 in dataIdList if not self.config.reprocessing or
                                      patchReprocessing[dataId1["patch"]]])

        # Remove persisted reprocessing determination
        if self.config.reprocessing:
            for patchId in patchReprocessing:
                if not patchReprocessing[patchId]:
                    continue
                dataId = dict(tract=tract, patch=patchId)
                filename = butler.get(
                    reprocessDataset + "_filename", dataId)[0]
                os.unlink(filename)

    def runDetection(self, cache, patchRef):
        """! Run detection on a patch

        Only slave nodes execute this method.

        @param cache: Pool cache, containing butler
        @param patchRef: Patch on which to do detection
        """
        with self.logOperation("do detections on {}".format(patchRef.dataId)):
            idFactory = self.detectCoaddSources.makeIdFactory(patchRef)
            coadd = patchRef.get(self.config.coaddName + "Coadd",
                                 immediate=True)
            expId = int(patchRef.get(self.config.coaddName + "CoaddId"))
            self.detectCoaddSources.emptyMetadata()
            detResults = self.detectCoaddSources.run(coadd, idFactory, expId=expId)
            self.detectCoaddSources.write(detResults, patchRef)
            self.detectCoaddSources.writeMetadata(patchRef)

    def runMergeDetections(self, cache, dataIdList):
        """!Run detection merging on a patch

        Only slave nodes execute this method.

        @param cache: Pool cache, containing butler
        @param dataIdList: List of data identifiers for the patch in different filters
        """
        with self.logOperation("merge detections from %s" % (dataIdList,)):
            dataRefList = [getDataRef(cache.butler, dataId, self.config.coaddName + "Coadd_calexp") for
                           dataId in dataIdList]
            if ("mergeCoaddDetections" in self.reuse and
                    dataRefList[0].datasetExists(self.config.coaddName + "Coadd_mergeDet", write=True)):
                self.log.info("Skipping mergeCoaddDetections for %s; output already exists." %
                              dataRefList[0].dataId)
                return
            self.mergeCoaddDetections.runDataRef(dataRefList)

    def runDeblendMerged(self, cache, dataIdList):
        """Run the deblender on a list of dataId's

        Only slave nodes execute this method.

        Parameters
        ----------
        cache: Pool cache
            Pool cache with butler.
        dataIdList: list
            Data identifier for patch in each band.

        Returns
        -------
        result: bool
            whether the patch requires reprocessing.
        """
        with self.logOperation("deblending %s" % (dataIdList,)):
            dataRefList = [getDataRef(cache.butler, dataId, self.config.coaddName + "Coadd_calexp") for
                           dataId in dataIdList]
            reprocessing = False  # Does this patch require reprocessing?
            if ("deblendCoaddSources" in self.reuse and
                all([dataRef.datasetExists(self.config.coaddName + "Coadd_" + self.measurementInput,
                                           write=True) for dataRef in dataRefList])):
                if not self.config.reprocessing:
                    self.log.info("Skipping deblendCoaddSources for %s; output already exists" % dataIdList)
                    return False

                # Footprints are the same every band, therefore we can check just one
                catalog = dataRefList[0].get(self.config.coaddName + "Coadd_" + self.measurementInput)
                bigFlag = catalog["deblend_parentTooBig"]
                # Footprints marked too large by the previous deblender run
                numOldBig = bigFlag.sum()
                if numOldBig == 0:
                    self.log.info("No large footprints in %s" % (dataRefList[0].dataId))
                    return False

                # This if-statement can be removed after DM-15662
                if self.config.deblendCoaddSources.simultaneous:
                    deblender = self.deblendCoaddSources.multiBandDeblend
                else:
                    deblender = self.deblendCoaddSources.singleBandDeblend

                # isLargeFootprint() can potentially return False for a source that is marked
                # too big in the catalog, because of "new"/different deblender configs.
                # numNewBig is the number of footprints that *will* be too big if reprocessed
                numNewBig = sum((deblender.isLargeFootprint(src.getFootprint()) for
                                 src in catalog[bigFlag]))
                if numNewBig == numOldBig:
                    self.log.info("All %d formerly large footprints continue to be large in %s" %
                                  (numOldBig, dataRefList[0].dataId,))
                    return False
                self.log.info("Found %d large footprints to be reprocessed in %s" %
                              (numOldBig - numNewBig, [dataRef.dataId for dataRef in dataRefList]))
                reprocessing = True

            self.deblendCoaddSources.runDataRef(dataRefList)
            return reprocessing

    def runMeasurements(self, cache, dataId):
        """Run measurement on a patch for a single filter

        Only slave nodes execute this method.

        Parameters
        ----------
        cache: Pool cache
            Pool cache, with butler
        dataId: dataRef
            Data identifier for patch
        """
        with self.logOperation("measurements on %s" % (dataId,)):
            dataRef = getDataRef(cache.butler, dataId,
                                 self.config.coaddName + "Coadd_calexp")
            if ("measureCoaddSources" in self.reuse and
                not self.config.reprocessing and
                    dataRef.datasetExists(self.config.coaddName + "Coadd_meas", write=True)):
                self.log.info("Skipping measuretCoaddSources for %s; output already exists" % dataId)
                return
            self.measureCoaddSources.runDataRef(dataRef)

    def runMergeMeasurements(self, cache, dataIdList):
        """!Run measurement merging on a patch

        Only slave nodes execute this method.

        @param cache: Pool cache, containing butler
        @param dataIdList: List of data identifiers for the patch in different filters
        """
        with self.logOperation("merge measurements from %s" % (dataIdList,)):
            dataRefList = [getDataRef(cache.butler, dataId, self.config.coaddName + "Coadd_calexp") for
                           dataId in dataIdList]
            if ("mergeCoaddMeasurements" in self.reuse and
                not self.config.reprocessing and
                    dataRefList[0].datasetExists(self.config.coaddName + "Coadd_ref", write=True)):
                self.log.info("Skipping mergeCoaddMeasurements for %s; output already exists" %
                              dataRefList[0].dataId)
                return
            self.mergeCoaddMeasurements.runDataRef(dataRefList)

    def runForcedPhot(self, cache, dataId):
        """!Run forced photometry on a patch for a single filter

        Only slave nodes execute this method.

        @param cache: Pool cache, with butler
        @param dataId: Data identifier for patch
        """
        with self.logOperation("forced photometry on %s" % (dataId,)):
            dataRef = getDataRef(cache.butler, dataId,
                                 self.config.coaddName + "Coadd_calexp")
            if ("forcedPhotCoadd" in self.reuse and
                not self.config.reprocessing and
                    dataRef.datasetExists(self.config.coaddName + "Coadd_forced_src", write=True)):
                self.log.info("Skipping forcedPhotCoadd for %s; output already exists" % dataId)
                return
            self.forcedPhotCoadd.runDataRef(dataRef)

    def writeMetadata(self, dataRef):
        """We don't collect any metadata, so skip"""
        pass
