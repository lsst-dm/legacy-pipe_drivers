import math
import collections

from .utils import ButlerTaskRunner, getDataRef
from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.pool import abortOnError, Pool, Debugger
from lsst.ctrl.pool.parallel import BatchPoolTask

Debugger().enabled = True


class SingleFrameDriverConfig(Config):
    processCcd = ConfigurableField(target=ProcessCcdTask, doc="CCD processing task")
    ignoreCcdList = ListField(dtype=int, default=[], doc="List of CCDs to ignore when processing")
    ccdKey = Field(dtype=str, default='', doc="DataId key corresponding to a single sensor i.e. in hsc 'ccd'")


class SingleFrameDriverTask(BatchPoolTask):
    """Process an entire exposure at once.

    We use MPI to gather the match lists for exposure-wide astrometric and
    photometric solutions.  Note that because of this, different nodes
    see different parts of the code.
    """

    RunnerClass = ButlerTaskRunner
    ConfigClass = SingleFrameDriverConfig
    _DefaultName = "singleFrameDriver"

    def __init__(self, *args, **kwargs):
        """Constructor.

        All nodes execute this method.
        """
        super(SingleFrameDriverTask, self).__init__(*args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processCcd")

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCpus):
        """Return walltime request for batch job

        Subclasses should override if the walltime should be calculated
        differently (e.g., addition of some serial time).

        @param time: Requested time per iteration
        @param parsedCmd: Results of argument parsing
        @param numCores: Number of cores
        """
        numCcds = sum(1 for raft in parsedCmd.butler.get("camera") for ccd in raft)
        numCycles = int(math.ceil(numCcds/float(numCpus)))
        numExps = len(cls.RunnerClass.getTargetList(parsedCmd))
        return time*numExps*numCycles

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        # Pop doBatch keyword before passing it along to the argument parser
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="singleFrameDriver", *args, **kwargs)
        parser.add_id_argument("--id", datasetType="raw", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    @abortOnError
    def run(self, expRef, butler):
        """Process a single exposure, with scatter-gather-scatter using MPI.
        """
        pool = Pool("processExposure")
        pool.cacheClear()
        pool.storeSet(butler=butler)

        dataIdList = dict([(ccdRef.get("ccdExposureId"), ccdRef.dataId)
                           for ccdRef in expRef.subItems("ccd") if ccdRef.datasetExists("raw")])
        dataIdList = collections.OrderedDict(sorted(dataIdList.items()))

        # Scatter: process CCDs independently
        structList = pool.map(self.process, dataIdList.values())
        numGood = sum(1 for s in structList if s is not None)
        if numGood == 0:
            self.log.warn("All CCDs in exposure failed")
            return

    def process(self, cache, dataId):
        """Process a single CCD and save the results for a later write.
        @param[in] cache    mpi pool cache variable. Provides a container for passing around objects
        @param[in] dataId   Data id for the individual ccd being processed

        @param[out] True/None Return True if the processing succeeds, else return None

        Only slaves execute this method.
        """
        cache.result = None
        if self.config.ccdKey == '':
            raise ValueError("The config parameter ccdKey must be set")
        if dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" % (dataId,))
            return None
        dataRef = getDataRef(cache.butler, dataId)
        ccdId = dataRef.get("ccdExposureId")
        with self.logOperation("processing %s (ccdId=%d)" % (dataId, ccdId)):
            try:
                self.processCcd.run(dataRef)
            except Exception, e:
                self.log.warn("Failed to process %s: %s\n" % (dataId, e))
                import traceback
                traceback.print_exc()
                return None

            return True
