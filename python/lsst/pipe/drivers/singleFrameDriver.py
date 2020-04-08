from lsst.pipe.base import ArgumentParser, ButlerInitializedTaskRunner, ConfigDatasetType
from lsst.pipe.tasks.processCcd import ProcessCcdTask
from lsst.pipe.tasks.postprocess import WriteSourceTableTask, TransformSourceTableTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask, BatchTaskRunner


class SingleFrameDriverConfig(Config):
    processCcd = ConfigurableField(
        target=ProcessCcdTask, doc="CCD processing task")
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


class SingleFrameTaskRunner(BatchTaskRunner, ButlerInitializedTaskRunner):
    """Run batches, and initialize Task using a butler"""
    pass


class SingleFrameDriverTask(BatchParallelTask):
    """Process CCDs in parallel
    """
    ConfigClass = SingleFrameDriverConfig
    _DefaultName = "singleFrameDriver"
    RunnerClass = SingleFrameTaskRunner

    def __init__(self, butler=None, psfRefObjLoader=None, astromRefObjLoader=None, photoRefObjLoader=None,
                 *args, **kwargs):
        """!
        Constructor

        The psfRefObjLoader, astromRefObjLoader, photoRefObjLoader should
        be an instance of LoadReferenceObjectsTasks that supplies an external
        reference catalog. They may be None if the butler argument is
        provided or the particular reference catalog is not required.

        @param[in] butler  The butler is passed to the refObjLoader constructor in case it is
            needed.  Ignored if the refObjLoader argument provides a loader directly.
        @param[in] psfRefObjLoader  Reference catalog loader for PSF determination.
        @param[in] astromRefObjLoader  Reference catalog loader for astrometric calibration.
        @param[in] photoRefObjLoader Reference catalog loader for photometric calibration.
        @param[in,out] kwargs  other keyword arguments for lsst.ctrl.pool.BatchParallelTask
        """
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("processCcd", butler=butler, psfRefObjLoader=psfRefObjLoader,
                         astromRefObjLoader=astromRefObjLoader, photoRefObjLoader=photoRefObjLoader)
        if self.config.doMakeSourceTable:
            self.makeSubtask("writeSourceTable")
            self.makeSubtask("transformSourceTable")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="singleFrameDriver", *args, **kwargs)
        parser.add_id_argument("--id",
                               datasetType=ConfigDatasetType(
                                   name="processCcd.isr.datasetType"),
                               level="sensor",
                               help="data ID, e.g. --id visit=12345 ccd=67")
        return parser

    def runDataRef(self, sensorRef):
        """Process a single CCD, with scatter-gather-scatter using MPI.
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" %
                          (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            result = self.processCcd.runDataRef(sensorRef)
            if self.config.doMakeSourceTable:
                parquet = self.writeSourceTable.run(result.calibRes.sourceCat,
                                                    ccdVisitId=sensorRef.get('ccdExposureId'))
                if self.config.doSaveWideSourceTable:
                    sensorRef.put(parquet.table, 'source')

                df = self.transformSourceTable.run(parquet.table,
                                                   funcs=self.transformSourceTable.getFunctors(),
                                                   dataId=sensorRef.dataId)
                self.transformSourceTable.write(df, sensorRef)

        return result
