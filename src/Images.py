from keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import numpy as np
import os

class Images:
    def __init__(self):
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
        allImages = np.array([])
        for filePath in filePathArray:
            # image = None
            # imageArray = None
            # if colorMode == 'RGB':
            #     image = Image.open(filePath).resize(targetSize)
            #     imageArray = np.asarray(image.getdata(), 'uint8').reshape(targetSize[0], targetSize[1], 3)
            # elif colorMode == 'L':
            #     image = Image.open(filePath).convert('L').resize(targetSize)
            #     imageArray = np.asarray(image.getdata(), 'uint8').reshape(targetSize[0], targetSize[1])
            image = Image.open(filePath).resize(targetSize)
            imageArray = np.asarray(image.getdata(), 'uint8').reshape(targetSize[0], targetSize[1], 3)
            allImages = np.append(allImages, imageArray)
        return allImages

    # Get image generator.
    def getImageGenerator(self, xArray, yArray, batchSize):
        imageGenerator = self.imgDataGen.flow(xArray, yArray, shuffle=True, batch_size=batchSize)
        return imageGenerator
