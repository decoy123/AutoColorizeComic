from keras.models import load_model
from PIL import Image
from src.Images import Images
import configparser
import numpy as np

# Get config.
config = configparser.ConfigParser()
config.read('./conf/config.ini')
# dataDir = sys.argv[1]
dataDir = config['COMMON']['DATA_DIR']
testDir = dataDir + '/' + config['TEST']['DIR']
testXDir = testDir + '/' + config['COMMON']['X_DIR']
testYDir = testDir + '/' + config['COMMON']['Y_DIR']
testBatchSize = int(config['TEST']['BATCH_SIZE'])
targetSize = (int(config['COMMON']['TARGET_SIZE_X']), int(config['COMMON']['TARGET_SIZE_Y']))
outDirectory = config['COMMON']['OUT_DIR']
modelFile = config['COMMON']['MODEL_FILE']

# Load image.
image = Images()
print('Start loading testX.')
testX = image.loadImages(testXDir, targetSize, 'L')
print('Start loading testY.')
testY = image.loadImages(testYDir, targetSize, 'RGB')
print('Start generating test.')
testGenerator = image.getImageGenerator(testX, testY, testBatchSize)

# Load model.
print('Start loading model.')
model = load_model(outDirectory + '/' + modelFile)

# Predict
print('Start predicting.')
epochs = int(config['COMMON']['EPOCHS'])
result = model.predict_generator(testGenerator, steps=epochs)
print(len(result))
for i, array in enumerate(result):
    pilImg = Image.fromarray(np.uint8(array))
    pilImg.save(config['COMMON']['OUT_DIR'] + '/' + str(i) + '.png', 'png')

print('End.')