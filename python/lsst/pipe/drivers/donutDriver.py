from __future__ import absolute_import, division, print_function

from lsst.pipe.base import ArgumentParser, ButlerInitializedTaskRunner, ConfigDatasetType
from lsst.pipe.tasks.processDonut import ProcessDonutTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask, BatchTaskRunner


class DonutDriverConfig(Config):
    processDonut = ConfigurableField(
        target=ProcessDonutTask, doc="Donut processing task")
    ignoreCcdList = ListField(dtype=int, default=[],
                              doc="List of CCDs to ignore when processing")
    ccdKey = Field(dtype=str, default="ccd",
                   doc="DataId key corresponding to a single sensor")


class DonutTaskRunner(BatchTaskRunner, ButlerInitializedTaskRunner):
    """Run batches, and initialize Task using a butler"""
    pass


class DonutDriverTask(BatchParallelTask):
    """Eat many donuts in parallel
    """
    ConfigClass = DonutDriverConfig
    _DefaultName = "donutDriver"
    RunnerClass = DonutTaskRunner

    def __init__(self, butler=None, psfRefObjLoader=None, *args, **kwargs):
        """!
        Constructor

        The psfRefObjLoader should be an instance of LoadReferenceObjectsTasks that supplies
        an external reference catalog. They may be None if the butler argument is provided
        or the particular reference catalog is not required.

        @param[in] butler  The butler is passed to the refObjLoader constructor in case it is
            needed.  Ignored if the refObjLoader argument provides a loader directly.
        @param[in] psfRefObjLoader  Reference catalog loader for PSF determination.
        @param[in,out] kwargs  other keyword arguments for lsst.ctrl.pool.BatchParallelTask
        """
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processDonut", butler=butler, psfRefObjLoader=psfRefObjLoader)

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="donutDriver", *args, **kwargs)
        parser.add_id_argument("--id",
                               datasetType=ConfigDatasetType(
                                   name="processDonut.isr.datasetType"),
                               level="sensor",
                               help="data ID, e.g. --id visit=12345 ccd=67")
        return parser

    def run(self, sensorRef):
        """Process a single CCD box of donuts, with scatter-gather-scatter using MPI.
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" %
                          (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            return self.processDonut.run(sensorRef)
