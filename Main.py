from keras.layers.core import *
from keras.layers.convolutional import *
from keras.models import Sequential
from PIL import Image
from src.Images import Images
import configparser

# Get config.
config = configparser.ConfigParser()
config.read('./conf/config.ini')
# Load image.
# dataDir = sys.argv[1]
dataDir = config['COMMON']['DATA_DIR']
trainDir = dataDir + '/' + config['TRAIN']['DIR']
validDir = dataDir + '/' + config['VALID']['DIR']
testDir = dataDir + '/' + config['TEST']['DIR']
trainXDir = trainDir + '/' + config['COMMON']['X_DIR']
trainYDir = trainDir + '/' + config['COMMON']['Y_DIR']
validXDir = validDir + '/' + config['COMMON']['X_DIR']
validYDir = validDir + '/' + config['COMMON']['Y_DIR']
testXDir = testDir + '/' + config['COMMON']['X_DIR']
testYDir = testDir + '/' + config['COMMON']['Y_DIR']

trainBatchSize = int(config['TRAIN']['BATCH_SIZE'])
validBatchSize = int(config['VALID']['BATCH_SIZE'])
testBatchSize = int(config['TEST']['BATCH_SIZE'])
targetSize = (int(config['COMMON']['TARGET_SIZE_X']), int(config['COMMON']['TARGET_SIZE_Y']))

image = Images()

print('Start loading trainX.')
trainX = image.loadImages(trainXDir, targetSize, 'L')
print('Start loading trainY.')
trainY = image.loadImages(trainYDir, targetSize, 'RGB')
print('Start loading validX.')
validX = image.loadImages(validXDir, targetSize, 'L')
print('Start loading validY.')
validY = image.loadImages(validYDir, targetSize, 'RGB')
print('Start loading testX.')
testX = image.loadImages(testXDir, targetSize, 'L')
print('Start loading testY.')
testY = image.loadImages(testYDir, targetSize, 'RGB')

print('Start generating train.')
trainGenerator = image.getImageGenerator(trainX, trainY, trainBatchSize)
print('Start generating valid.')
validGenerator = image.getImageGenerator(validX, validY, validBatchSize)
print('Start generating test.')
testGenerator = image.getImageGenerator(testX, testY, testBatchSize)

# Create model.
print('Start creating model.')
inputChannels = 1
# inputShape = (trainBatchSize, targetSize[0], targetSize[1], inputChannels)
inputShape = (targetSize[0], targetSize[1], inputChannels)

model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=inputShape))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(UpSampling2D(size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(3, (1, 1), activation='sigmoid'))

# Compile
print('Start compiling.')
model.compile(loss='mean_squared_error', optimizer='adagrad', metrics=['accuracy'])

# Fit
print('Start fitting.')
epochs = int(config['COMMON']['EPOCHS'])
model.fit_generator(trainGenerator,
                    steps_per_epoch=trainBatchSize,
                    epochs=epochs,
                    validation_data=validGenerator,
                    validation_steps=validBatchSize)

# Evaluate
print('Start evaluating.')
model.evaluate_generator(testGenerator, steps=testBatchSize)
result = model.predict_generator(testGenerator, steps=testBatchSize)
print(len(result))
for i, array in enumerate(result):
    arrayRgb = np.dstack((array[0], array[1], array[2]))
    arrayRgb255 = np.round(arrayRgb * 255)
    pilImg = Image.fromarray(np.uint8(arrayRgb255))
    pilImg.save(config['COMMON']['OUT_DIR'] + '/' + str(i) + '.png', 'png')

print('End.')