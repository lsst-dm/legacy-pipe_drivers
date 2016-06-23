from lsst.pipe.base import ArgumentParser, ButlerInitializedTaskRunner
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask

class SingleFrameDriverConfig(Config):
    processCcd = ConfigurableField(target=ProcessCcdTask, doc="CCD processing task")
    ignoreCcdList = ListField(dtype=int, default=[], doc="List of CCDs to ignore when processing")
    ccdKey = Field(dtype=str, default="ccd", doc="DataId key corresponding to a single sensor")


class SingleFrameDriverTask(BatchParallelTask):
    """Process CCDs in parallel
    """
    ConfigClass = SingleFrameDriverConfig
    _DefaultName = "singleFrameDriver"
    RunnerClass = ButlerInitializedTaskRunner

    def __init__(self, butler=None, *args, **kwargs):
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processCcd", butler=butler)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="singleFrameDriver", *args, **kwargs)
        parser.add_id_argument("--id", datasetType="raw", level="sensor",
                               help="data ID, e.g. --id visit=12345 ccd=67")
        return parser

    def run(self, sensorRef):
        """Process a single CCD, with scatter-gather-scatter using MPI.
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" % (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            return self.processCcd.run(sensorRef)
