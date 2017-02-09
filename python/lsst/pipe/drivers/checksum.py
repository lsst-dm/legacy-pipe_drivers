
import hashlib
import zlib
import cPickle as pickle

import lsst.afw.image as afwImage

__all__ = ["checksum", ]

# Image types to support
exposureTypes = (afwImage.ExposureF, afwImage.ExposureD,)
maskedImageTypes = (afwImage.MaskedImageF, afwImage.MaskedImageD,)
decoratedImageTypes = (afwImage.DecoratedImageF, afwImage.DecoratedImageD,)
imageTypes = (afwImage.ImageF, afwImage.ImageD, afwImage.ImageI,)

PROTOCOL = 2  # Pickling protocol

# Functions for creating the checksum
sumFunctions = {
    "CRC32": lambda obj: zlib.crc32(pickle.dumps(obj, PROTOCOL)),
    "MD5": lambda obj: hashlib.md5(pickle.dumps(obj, PROTOCOL)).hexdigest(),
}


def checksum(obj, header=None, sumType="MD5"):
    """!Calculate a checksum of an object

    We have special handling for images (e.g., breaking a MaskedImage into
    its various components), but the object may be any picklable type.

    @param obj  Object for which to calculate the checksum
    @param header  FITS header (PropertyList) to update with checksum values, or None
    @param sumType  Type of checksum to calculate
    @return dict with header keyword,value pairs
    """
    assert sumType in sumFunctions, "Unknown sumType: %s" % (sumType,)
    func = sumFunctions[sumType]

    results = {}

    if isinstance(obj, exposureTypes):
        obj = obj.getMaskedImage()
    if isinstance(obj, decoratedImageTypes):
        obj = obj.getImage()

    if isinstance(obj, maskedImageTypes):
        results[sumType + "_IMAGE"] = func(obj.getImage())
        results[sumType + "_MASK"] = func(obj.getMask())
        results[sumType + "_VARIANCE"] = func(obj.getVariance())
    elif isinstance(obj, imageTypes):
        results[sumType + "_IMAGE"] = func(obj)
    else:
        results[sumType] = func(obj)

    if header is not None:
        for k, v in results.iteritems():
            header.add(k, v)

    return results
