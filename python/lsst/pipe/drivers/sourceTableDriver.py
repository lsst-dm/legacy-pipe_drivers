from lsst.pipe.base import ArgumentParser
from lsst.pipe.tasks.postprocess import WriteSourceTableTask, TransformSourceTableTask
from lsst.pex.config import Config, Field, ConfigurableField, ListField
from lsst.ctrl.pool.parallel import BatchParallelTask, BatchTaskRunner


class SourceTableDriverConfig(Config):
    writeSourceTable = ConfigurableField(
        target=WriteSourceTableTask,
        doc="Task to make parquet table for full src catalog",
    )
    transformSourceTable = ConfigurableField(
        target=TransformSourceTableTask,
        doc="Transform Source Table to DPDD specification",
    )
    ignoreCcdList = ListField(
        dtype=int,
        default=[],
        doc="List of CCDs to ignore when processing",
    )
    ccdKey = Field(
        dtype=str,
        default="ccd",
        doc="DataId key corresponding to a single sensor",
    )

    def setDefaults(self):
        self.writeSourceTable.doApplyExternalPhotoCalib=True
        self.writeSourceTable.doApplyExternalSkyWcs=True


class SourceTableDriverTask(BatchParallelTask):
    """Convert existing src tables to parquet Source Table

    This driver can convert PDR2-era `src` tables that do not have
    * local photo calib columns
    * local wcs columns
    * sky_source flag and
    * detect_isPrimary flags set.set

    It is specialized for the 2021 HSC data release in which we will
    not rerun singleFrameDriver but start the processing from FGCM.

    Can be removed during after the Gen2 deprecation period.
    """
    ConfigClass = SourceTableDriverConfig
    _DefaultName = "sourceTableDriver"
    RunnerClass = BatchTaskRunner

    def __init__(self, *args, **kwargs):
        """!
        Constructor

        @param[in,out] kwargs  other keyword arguments for lsst.ctrl.pool.BatchParallelTask
        """
        BatchParallelTask.__init__(self, *args, **kwargs)
        self.ignoreCcds = set(self.config.ignoreCcdList)
        self.makeSubtask("writeSourceTable")
        self.makeSubtask("transformSourceTable")

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="sourceTableDriver", *args, **kwargs)
        parser.add_id_argument("--id",
                               datasetType="src",
                               level="sensor",
                               help="data ID, e.g. --id visit=12345 ccd=67")
        return parser

    def runDataRef(self, sensorRef):
        """Process a single CCD
        """
        if sensorRef.dataId[self.config.ccdKey] in self.ignoreCcds:
            self.log.warn("Ignoring %s: CCD in ignoreCcdList" %
                          (sensorRef.dataId))
            return None

        with self.logOperation("processing %s" % (sensorRef.dataId,)):
            res = self.writeSourceTable.runDataRef(sensorRef)
            df = self.transformSourceTable.run(res.table,
                                               funcs=self.transformSourceTable.getFunctors(),
                                               dataId=sensorRef.dataId)
            return df

    def _getMetadataName(self):
        """There's no metadata to write out"""
        return None
