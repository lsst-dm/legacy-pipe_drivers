from lsst.ctrl.pool.pool import Pool, startPool, abortOnError
from lsst.ctrl.pool.parallel import BatchCmdLineTask
from lsst.pipe.base import Struct
from lsst.pipe.tasks.ingest import IngestTask, IngestError


class PoolIngestTask(BatchCmdLineTask, IngestTask):
    """Parallel version of IngestTask"""
    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        return float(time)*len(parsedCmd.files)/numCores

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        """Build an ArgumentParser

        Removes the batch-specific parts.
        """
        kwargs.pop("doBatch", False)
        kwargs.pop("add_help", False)
        return cls.ArgumentParser(*args, name="ingest", **kwargs)

    @classmethod
    def parseAndRun(cls, *args, **kwargs):
        """Run with a MPI process pool"""
        pool = startPool()
        config = cls.ConfigClass()
        parser = cls.ArgumentParser(name=cls._DefaultName)
        args = parser.parse_args(config)
        task = cls(config=args.config)
        task.run(args)
        pool.exit()

    def runFileWrapper(self, struct, args):
        """Run ingest on one file

        This is a wrapper method for calling ``runFile``.

        Parameters
        ----------
        struct : `lsst.pipe.base.Struct`
            Structure containing ``filename`` (`str`) and ``position`` (`int`).
        args : `argparse.Namespace`
            Parsed command-line arguments.

        Returns
        -------
        hduInfoList : `list` of `dict`
            Parsed information from FITS HDUs, or ``None``.
        """
        filename = struct.filename
        position = struct.position
        try:
            return self.runFile(filename, None, args, position)
        except IngestError as exc:
            self.log.warn(f"Unable to ingest {filename}: {exc}")
            return None

    @abortOnError
    def run(self, args):
        """Run ingest

        We read and ingest the files in parallel, and then
        stuff the registry database in serial.
        """
        # Parallel
        pool = Pool(None)
        filenameList = self.expandFiles(args.files)
        dataList = [Struct(filename=filename, position=ii) for ii, filename in enumerate(filenameList)]
        infoList = pool.map(self.runFileWrapper, dataList, args)

        # Serial
        root = args.input
        context = self.register.openRegistry(root, create=args.create, dryrun=args.dryrun)
        with context as registry:
            for hduInfoList in infoList:
                if hduInfoList is None:
                    continue
                for info in hduInfoList:
                    self.register.addRow(registry, info, dryrun=args.dryrun, create=args.create)

    def writeConfig(self, *args, **kwargs):
        pass

    def writeMetadata(self, *args, **kwargs):
        pass
