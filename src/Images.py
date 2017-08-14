from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os

class Images():
    def __init__(self):
        super().__init__()
        self.imgDataGen = ImageDataGenerator(
            # featurewise_center=True,
            # samplewise_center=True,
            # featurewise_std_normalization=True,
            # samplewise_std_normalization=True,
            rescale=1. / 255
        )

    # Get image matrix.
    def loadImages(self, directory, targetSize, colorMode):
        # Get file paths in the directory.
        filePathArray = []
        with os.scandir(directory) as it:
            for entry in it:
                if entry.is_file():
                    filePathArray.append(entry.path)
        # Get binary of images.
        colorDimension = None
        if colorMode == 'RGB':
            colorDimension = 3
        elif colorMode == 'L':
            colorDimension = 1
        allImages = []
        for filePath in filePathArray:
            image = Image.open(filePath).resize(targetSize)
            if colorMode == 'L':
                image = image.convert(colorMode)
            imageArray = np.asarray(image.getdata(), 'uint8').reshape(targetSize[0], targetSize[1], colorDimension)
            allImages.append(imageArray)
        return np.array(allImages)

    # Get image generator.
    def getImageGenerator(self, xArray, yArray, batchSize):
        imageGenerator = self.imgDataGen.flow(xArray, yArray, shuffle=True, batch_size=batchSize)
        return imageGenerator
