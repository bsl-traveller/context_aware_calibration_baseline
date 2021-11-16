# Constants
NUM_CLASSES = 10
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS = 3
BATCH_SIZE = 64
EPOCHS = 350
WEIGHT_DECAY = 0.0001
SEEDS = [32, 100, 242, 376, 498]
#SEEDS = [32]
FILTER_SAMPLES = 10 #samples per distortions
PATIENCE = 100
LABEL_SMOOTH = 0.0 #0.0, 0.1 or 0.2
CONTEXTS = 6 #original, blur, detail, edge_enhance, smooth, sharp
