# PROJECT
ARCHITECTURES = ['avr', 'alphaev56', 'arm', 'm68k', 'mips', 'mipsel', 'powerpc', 's390', 'sh4', 'sparc', 'x86_64', 'xtensa']
DATA_PATH = 'bdata.npy'
LABELS_PATH = 'labels.npy'
MODEL_PATH = 'bcmodel.pt'

# MODEL PARAMS
LEARNING_RATE = 1e-3
NUM_HIDDEN_LAYERS = 3
INPUT_DIM = 64
HIDDEN_DIM = 256
OUTPUT_DIM = len(ARCHITECTURES)

# TRAINING PARAMS
NUM_SAMPLES = 5000
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.9
BATCH_SIZE = 4
MAX_EPOCHS = 50
PATIENCE = 7

# USEFUL CONSTRUCTS
LABELS = {arch: i for i, arch in enumerate(ARCHITECTURES)}
PRED_LABELS = {i: arch for i, arch in enumerate(ARCHITECTURES)}