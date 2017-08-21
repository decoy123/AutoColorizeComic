from keras.layers.convolutional import *
from keras.models import Sequential
from src.Images import Images
import configparser

# Get config.
config = configparser.ConfigParser()
config.read('./conf/config.ini')
# dataDir = sys.argv[1]
dataDir = config['COMMON']['DATA_DIR']
trainDir = dataDir + '/' + config['TRAIN']['DIR']
validDir = dataDir + '/' + config['VALID']['DIR']
trainXDir = trainDir + '/' + config['COMMON']['X_DIR']
trainYDir = trainDir + '/' + config['COMMON']['Y_DIR']
validXDir = validDir + '/' + config['COMMON']['X_DIR']
validYDir = validDir + '/' + config['COMMON']['Y_DIR']
trainBatchSize = int(config['TRAIN']['BATCH_SIZE'])
validBatchSize = int(config['VALID']['BATCH_SIZE'])
targetSize = (int(config['COMMON']['TARGET_SIZE_X']), int(config['COMMON']['TARGET_SIZE_Y']))
outDirectory = config['COMMON']['OUT_DIR']
modelFile = config['COMMON']['MODEL_FILE']

# Load image.
image = Images()
print('Start loading trainX.')
trainX = image.loadImages(trainXDir, targetSize, 'L')
print('Start loading trainY.')
trainY = image.loadImages(trainYDir, targetSize, 'RGB')
print('Start loading validX.')
validX = image.loadImages(validXDir, targetSize, 'L')
print('Start loading validY.')
validY = image.loadImages(validYDir, targetSize, 'RGB')
print('Start generating train.')
trainGenerator = image.getImageGenerator(trainX, trainY, trainBatchSize)
print('Start generating valid.')
validGenerator = image.getImageGenerator(validX, validY, validBatchSize)

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

# Save model.
print('Start saving model.')
model.save(outDirectory + '/' + modelFile)

print('End.')