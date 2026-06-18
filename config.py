# PROJECT
ARCHITECTURES = ['avr', 'alphaev56', 'arm', 'm68k', 'mips', 'mipsel', 'powerpc', 's390', 'sh4', 'sparc', 'x86_64', 'xtensa']
DATA_PATH = 'bdata.npy'
LABELS_PATH = 'labels.npy'
MODEL_PATH = 'bcmodel.pt'

# MODEL PARAMS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.4
NUM_HIDDEN_LAYERS = 3
EMBEDDING_DIM = 32
POOL_MOD = 4
INPUT_DIM = (3+POOL_MOD) * EMBEDDING_DIM   # 3 for mean-max-std pooling
HIDDEN_DIM = 256
OUTPUT_DIM = len(ARCHITECTURES)

# TRAINING PARAMS
BATCH_SIZE = 32
NUM_SAMPLES = 32 * 200  # 6400
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.9
MAX_EPOCHS = 50
PATIENCE = 7

# USEFUL CONSTRUCTS
LABELS = {arch: i for i, arch in enumerate(ARCHITECTURES)}
PRED_LABELS = {i: arch for i, arch in enumerate(ARCHITECTURES)}