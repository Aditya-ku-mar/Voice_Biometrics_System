class Config:
    # Audio Parameters
    SAMPLE_RATE = 16000         
    NUM_MELS = 80               
    FRAME_LENGTH = 25           
    FRAME_SHIFT = 10             
    MAX_AUDIO_LENGTH = 3.0      
    
    # Model Parameters
    INPUT_CHANNELS = 80
    CHANNELS = 512
    EMBEDDING_SIZE = 192
    SCALE = 8
    KERNEL_SIZE = 3
    DILATIONS = [2, 3, 4]

    # Training Parameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 2e-5
    NUM_WORKERS = 4
    DEVICE = "cuda"
    
    # AAM Softmax
    MARGIN = 0.2
    SCALE_FACTOR = 30

    # Random Seed
    SEED = 42
    CHECKPOINT_DIR = "checkpoints"
    LOG_DIR = "logs"