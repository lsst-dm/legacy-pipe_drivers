from __future__ import absolute_import, division, print_function

import numpy as np

import lsst.afw.math as afwMath
import lsst.afw.image as afwImage

from lsst.afw.cameraGeom.utils import makeImageFromCamera
from lsst.pipe.base import ArgumentParser
from lsst.pex.config import Config, Field
from lsst.ctrl.pool.pool import Pool
from lsst.ctrl.pool.parallel import BatchPoolTask


def makeCameraImage(camera, exposures, binning):
    """Make and write an image of an entire focal plane

    Parameters
    ----------
    camera : `lsst.afw.cameraGeom.Camera`
        Camera description.
    exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
        CCD exposures, binned by `binning`.
    binning : `int`
        Binning size that has been applied to images.
    """
    class ImageSource(object):
        """Source of images for makeImageFromCamera"""
        def __init__(self, exposures):
            """Constructor

            Parameters
            ----------
            exposures : `dict` mapping detector ID to `lsst.afw.image.Exposure`
                CCD exposures, already binned.
            """
            self.isTrimmed = True
            self.exposures = exposures
            self.background = np.nan

        def getCcdImage(self, detector, imageFactory, binSize):
            """Provide image of CCD to makeImageFromCamera"""
            detId = detector.getId()
            if detId not in self.exposures:
                dims = detector.getBBox().getDimensions()/binSize
                image = imageFactory(*[int(xx) for xx in dims])
                image.set(self.background)
            else:
                image = self.exposures[detector.getId()]
            if hasattr(image, "getMaskedImage"):
                image = image.getMaskedImage()
            if hasattr(image, "getMask"):
                mask = image.getMask()
                isBad = mask.getArray() & mask.getPlaneBitMask("NO_DATA") > 0
                image = image.clone()
                image.getImage().getArray()[isBad] = self.background
            if hasattr(image, "getImage"):
                image = image.getImage()
            return image, detector

    image = makeImageFromCamera(
        camera,
        imageSource=ImageSource(exposures),
        imageFactory=afwImage.ImageF,
        binSize=binning
    )
    return image


class VisualizeVisitConfig(Config):
    binning = Field(dtype=int, default=8, doc="Binning factor to apply")


class VisualizeVisitTask(BatchPoolTask):
    ConfigClass = VisualizeVisitConfig
    _DefaultName = "visualizeVisit"

    def __init__(self, *args, **kwargs):
        BatchPoolTask.__init__(self, *args, **kwargs)
        self._storedButler = False  # Stored butler in the Pool? Doing this once increases efficiency

    @classmethod
    def _makeArgumentParser(cls, *args, **kwargs):
        kwargs.pop("doBatch", False)
        parser = ArgumentParser(name="visualizeVisit", *args, **kwargs)
        parser.add_id_argument("--id", datasetType="calexp", level="visit",
                               help="data ID, e.g. --id visit=12345")
        return parser

    @classmethod
    def batchWallTime(cls, time, parsedCmd, numCores):
        """Return walltime request for batch job

        Subclasses should override if the walltime should be calculated
        differently (e.g., addition of some serial time).

        Parameters
        ----------
        time : `float`
            Requested time per iteration.
        parsedCmd : `argparse.Namespace`
            Results of argument parsing.
        numCores : `int`
            Number of cores.
        """
        numTargets = len(cls.getTargetList(parsedCmd))
        return time*numTargets

    def runDataRef(self, expRef):
        """Generate an image of the entire visit

        Only the master node executes this method; it controls the slave nodes,
        which do the data retrieval.

        Parameters
        ----------
        expRef : `lsst.daf.persistence.ButlerDataRef`
            Data reference for exposure.
        """
        pool = Pool()

        if not self._storedButler:
            pool.storeSet(butler=expRef.getButler())

        with self.logOperation("processing %s" % (expRef.dataId,)):
            camera = expRef.get("camera")
            dataIdList = [ccdRef.dataId for ccdRef in expRef.subItems("ccd") if
                          ccdRef.datasetExists("calexp")]

            exposures = pool.map(self.readImage, dataIdList)
            exposures = dict(keyValue for keyValue in exposures if keyValue is not None)
            image = makeCameraImage(camera, exposures, self.config.binning)
            expRef.put(image, "calexp_camera")

    def readImage(self, cache, dataId):
        """Collect original image for visualisation

        This method runs on the slave nodes.

        Parameters
        ----------
        cache : `lsst.pipe.base.Struct`
            Process pool cache.
        dataId : `dict`
            Data identifier.

        Returns
        -------
        detId : `int`
            Detector identifier.
        image : `lsst.afw.image.MaskedImage`
            Binned image.
        """
        exposure = cache.butler.get("calexp", dataId)
        return (exposure.getDetector().getId(),
                afwMath.binImage(exposure.getMaskedImage(), self.config.binning))

    def _getConfigName(self):
        """It's not worth preserving the configuration"""
        return None

    def _getMetadataName(self):
        """There's no metadata to write out"""
        return None
