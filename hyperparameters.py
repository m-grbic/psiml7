# Hyperparameters
W1, W2 = 1, 1e-3 # Loss weights
LR = 1e-3
EPOCHS = 200
BATCH_SIZE = 64
GAMMA, STEP = 0.1, 150
USE_SCHEDULER = False
LOSS = 'l1'
SMOOTH_L1 = True
SMOOTH_THRESH = 0.5
MODEL_NAME = '2021_08_08_N08'

# Transformations
BORDER_SIZE = 10
IMG_HEIGHT = 96
IMG_WIDTH = 128
IMG_HEIGHT_RESCALE = 128
IMG_WIDTH_RESCALE = 150
P_FLIP = 0.5

# MixUp
MIXUP = False
MIXUP_SIZE = 0.4

# Blend
BLEND = False
BLEND_SIZE = 0.2

# Rotation
ROTATE = True
ROTATION_SIZE = 0.5
ROTATION_ANGLE = 10

# Train test split
SPLIT_RATIO = 0.1