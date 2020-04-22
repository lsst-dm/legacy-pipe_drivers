from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.processCcdWithFakes import ProcessCcdWithFakesTask
from lsst.pipe.tasks.postprocess import WriteSourceTableTask, TransformSourceTableTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask, BatchTaskRunner
from lsst.meas.base import PerTractCcdDataIdContainer


class ProcessCcdWithFakesDriverConfig(Config):
    processCcdWithFakes = ConfigurableField(
        target=ProcessCcdWithFakesTask, doc="CCD with fakes processing task")
    doMakeSourceTable = Field(dtype=bool, default=False,
                              doc="Do postprocessing tasks to write parquet Source Table?")
    doSaveWideSourceTable = Field(dtype=bool, default=False,
                                  doc=("Save the parquet version of the full src catalog?",
                                       "Only respected if doMakeSourceTable"))
    writeSourceTable = ConfigurableField(
        target=WriteSourceTableTask, doc="Task to make parquet table for full src catalog")
    transformSourceTable = ConfigurableField(
        target=TransformSourceTableTask, doc="Transform Source Table to DPDD specification")
    ignoreCcdList = ListField(dtype=int, default=[],
                              doc="List of CCDs to ignore when processing")
    ccdKey = Field(dtype=str, default="ccd",
                   doc="DataId key corresponding to a single sensor")


class ProcessCcdWithFakesTaskRunner(BatchTaskRunner):
    """Run batches, and initialize Task"""
    pass


class ProcessCcdWithFakesDriverTask(BatchParallelTask):
    """Process CCDs in parallel for processCcdWithFakes
    """
    ConfigClass = ProcessCcdWithFakesDriverConfig
    _DefaultName = "processCcdWithFakesDriver"
    RunnerClass = ProcessCcdWithFakesTaskRunner

    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        kwargs :  other keyword arguments for lsst.ctrl.pool.BatchParallelTask
        """
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processCcdWithFakes")
        if self.config.doMakeSourceTable:
            self.makeSubtask("writeSourceTable")
            self.makeSubtask("transformSourceTable")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="processCcdWithFakesDriver", *args, **kwargs)
        parser.add_id_argument("--id", "fakes_calexp",
                               help="data ID, e.g. --id visit=12345 ccd=67, tract=9813",
                               ContainerClass=PerTractCcdDataIdContainer)
        return parser

    def runDataRef(self, sensorRef):
        """Process a single CCD, with scatter-gather-scatter using MPI.
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" %
                          (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            result = self.processCcdWithFakes.runDataRef(sensorRef)
            if self.config.doMakeSourceTable:
                parquet = self.writeSourceTable.run(result.outputCat,
                                                    ccdVisitId=sensorRef.get('ccdExposureId'))
                if self.config.doSaveWideSourceTable:
                    sensorRef.put(parquet.table, 'fakes_source')

                df = self.transformSourceTable.run(parquet.table,
                                                   funcs=self.transformSourceTable.getFunctors(),
                                                   dataId=sensorRef.dataId)
                self.transformSourceTable.write(df, sensorRef)

        return result
