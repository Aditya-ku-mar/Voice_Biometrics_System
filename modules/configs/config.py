import torch
class Config:
    # Dataset
    TRAIN_PATH = "./data/LibriSpeech/train-clean-100"
    VAL_PATH = "./data/LibriSpeech/dev-clean"
    TEST_PATH = "./data/LibriSpeech/test-clean"

    SAMPLE_RATE = 16000
    SEGMENT_LENGTH = 3
    NUM_MELS = 80
    FRAME_LENGTH = 25
    FRAME_SHIFT = 10

    # Model
    INPUT_CHANNELS = 80
    CHANNELS = 512
    EMBEDDING_SIZE = 192
    RES2NET_SCALE = 8
    KERNEL_SIZE = 3
    DILATIONS = [2, 3, 4]

    # AAM Softmax
    MARGIN = 0.2
    AAM_SCALE = 30

    # Training
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 2e-5
    NUM_WORKERS = 4
    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    SEED = 42
    # Paths
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"