from yacs.config import CfgNode as CN

_CN = CN()

##############  ↓  RENI GLOBAL  ↓  ##############
_CN.RENI = CN()
_CN.RENI.TASKS = ["FIT_DECODER", "FIT_LATENT"] # FIT_DECODER, FIT_LATENT, FIT_INVERSE
_CN.RENI.MODEL_TYPE = "VariationalAutoDecoder" # AutoDecoder, VariationalAutoDecoder
_CN.RENI.CONDITIONING = "FiLM" # "FiLM" or "Cond-by-Concat"
_CN.RENI.EQUIVARIANCE = "SO2" # SO3 | SO2 | None
_CN.RENI.LATENT_DIMENSION = 36 # In the paper this is N : (D = N x 3)
_CN.RENI.HIDDEN_LAYERS = 5 # Number of hidden layers in the decoder
_CN.RENI.HIDDEN_FEATURES = 256 # Number of features in each hidden layer
_CN.RENI.OUT_FEATURES = 3 # Number of features in the output layer, RGB
_CN.RENI.LAST_LAYER_LINEAR = True # If True, the last layer is linear, else a SIREN layer
_CN.RENI.OUTPUT_ACTIVATION = None # "tanh" | "exp" | None
_CN.RENI.FIRST_OMEGA_0 = 30.0 
_CN.RENI.HIDDEN_OMEGA_0 = 30.0
_CN.RENI.MAPPING_LAYERS = 3 # Number of layers in the mapping network (if FiLM)
_CN.RENI.MAPPING_FEATURES = 256 # Number of features in each mapping layer (if FiLM)

#####  ↓  TASK SPECIFIC HYPERPARAMETERS  ↓  #####
##############  ↓  FIT_DECODER  ↓  ##############
_CN.RENI.FIT_DECODER = CN()
_CN.RENI.FIT_DECODER.LR_START = 1e-5
_CN.RENI.FIT_DECODER.LR_END = 1e-7 # If using lr scheduler, this will be the final lr
_CN.RENI.FIT_DECODER.OPTIMIZER = "adam"
_CN.RENI.FIT_DECODER.OPTIMIZER_BETA_1 = 0.0
_CN.RENI.FIT_DECODER.OPTIMIZER_BETA_2 = 0.999
_CN.RENI.FIT_DECODER.SCHEDULER_TYPE = "exponential"
_CN.RENI.FIT_DECODER.SCHEDULER_STEP_SIZE = 1
_CN.RENI.FIT_DECODER.SCHEDULER_GAMMA = 1
_CN.RENI.FIT_DECODER.BATCH_SIZE = 1
_CN.RENI.FIT_DECODER.EPOCHS = 2400
_CN.RENI.FIT_DECODER.MULTI_RES_TRAINING = True # If False, FINAL_RESOLUTION will be used
_CN.RENI.FIT_DECODER.INITAL_RESOLUTION = [16, 32]
_CN.RENI.FIT_DECODER.FINAL_RESOLUTION = [64, 128] # If MULTI_RES_TRAINING is False, this will be used
_CN.RENI.FIT_DECODER.CURRICULUM = [25, 80, 150] # Epochs at which to double the resolution [100, 500, ...] | None ... if None res changes will be evenly spaced across epochs
_CN.RENI.FIT_DECODER.KLD_WEIGHTING = 1e-4 # Weighting of the KLD loss

##############  ↓  FIT_LATENT  ↓  ##############
_CN.RENI.FIT_LATENT = CN()
_CN.RENI.FIT_LATENT.LR_START = 1e-2
_CN.RENI.FIT_LATENT.LR_END = 1e-5 # If using lr scheduler, this will be the final lr
_CN.RENI.FIT_LATENT.OPTIMIZER = "adam"
_CN.RENI.FIT_LATENT.OPTIMIZER_BETA_1 = 0.0
_CN.RENI.FIT_LATENT.OPTIMIZER_BETA_2 = 0.999
_CN.RENI.FIT_LATENT.SCHEDULER_TYPE = "exponential"
_CN.RENI.FIT_LATENT.SCHEDULER_STEP_SIZE = 1
_CN.RENI.FIT_LATENT.SCHEDULER_GAMMA = 1
_CN.RENI.FIT_LATENT.BATCH_SIZE = 1
_CN.RENI.FIT_LATENT.EPOCHS = 1200
_CN.RENI.FIT_LATENT.MULTI_RES_TRAINING = True # If False, FINAL_RESOLUTION will be used
_CN.RENI.FIT_LATENT.INITAL_RESOLUTION = [16, 32]
_CN.RENI.FIT_LATENT.FINAL_RESOLUTION = [64, 128] # If MULTI_RES_TRAINING is False, this will be used
_CN.RENI.FIT_LATENT.CURRICULUM = [25, 80, 150] # Epochs at which to double the resolution [100, 500, ...] | None ... if None will be evenly spaced across epochs
_CN.RENI.FIT_LATENT.COSINE_SIMILARITY_WEIGHT = 1e-4 # Weighting of the cosine similarity loss
_CN.RENI.FIT_LATENT.PRIOR_LOSS_WEIGHT = 1e-7 # Weighting of the prior loss
_CN.RENI.FIT_LATENT.APPLY_MASK = False # Masking for RENI inpainting task
_CN.RENI.FIT_LATENT.MASK_PATH = "data/processed/Masks/Mask-3.png" # Path to mask for RENI inpainting task

##############  ↓  FIT_INVERSE  ↓  ##############
_CN.RENI.FIT_INVERSE = CN()
_CN.RENI.FIT_INVERSE.LR_START = 1e-2
_CN.RENI.FIT_INVERSE.LR_END = 1e-5 # If using lr scheduler, this will be the final lr
_CN.RENI.FIT_INVERSE.OPTIMIZER = "adam"
_CN.RENI.FIT_INVERSE.OPTIMIZER_BETA_1 = 0.0
_CN.RENI.FIT_INVERSE.OPTIMIZER_BETA_2 = 0.999
_CN.RENI.FIT_INVERSE.SCHEDULER_TYPE = "exponential"
_CN.RENI.FIT_INVERSE.SCHEDULER_STEP_SIZE = 1
_CN.RENI.FIT_INVERSE.SCHEDULER_GAMMA = 1  
_CN.RENI.FIT_INVERSE.BATCH_SIZE = 1
_CN.RENI.FIT_INVERSE.EPOCHS = 1200
_CN.RENI.FIT_INVERSE.MULTI_RES_TRAINING = False # If False, FINAL_RESOLUTION will be used
_CN.RENI.FIT_INVERSE.INITAL_RESOLUTION = [16, 32]
_CN.RENI.FIT_INVERSE.FINAL_RESOLUTION = [64, 128] # If MULTI_RES_TRAINING is False, this will be used
_CN.RENI.FIT_INVERSE.CURRICULUM = [25, 80, 150] # Epochs at which to double the resolution [100, 500, ...] | None ... if None will be evenly spaced across epochs
_CN.RENI.FIT_INVERSE.COSINE_SIMILARITY_WEIGHT = 1e-4
_CN.RENI.FIT_INVERSE.PRIOR_LOSS_WEIGHT = 1e-7
_CN.RENI.FIT_INVERSE.RENDERER = "PyTorch3D" # Renderer to use for inverse rendering task
_CN.RENI.FIT_INVERSE.RENDER_RESOLUTION = 64 # Resolution at which to render the object
_CN.RENI.FIT_INVERSE.OBJECT_PATH = "data/graphics/3d_models/teapot.obj" # Path to object to render
_CN.RENI.FIT_INVERSE.KD_VALUE = 1.0 # Value of the diffuse term in blinn-phong shading, specular term is 1.0 - KD_VALUE

##############  ↓  DATASET  ↓  ##############
_CN.DATASET = CN()
_CN.DATASET.NAME = "RENI_HDR" # RENI_HDR | RENI_LDR | CUSTOM

_CN.DATASET.RENI_HDR = CN()
_CN.DATASET.RENI_HDR.PATH = "data/RENI_HDR"
_CN.DATASET.RENI_HDR.TRANSFORMS = [["minmaxormalise", [-18.0536, 11.4633]]] # resize and totensor applied automatically for minmaxnormalise arg provide [] if not known else [min, max]
_CN.DATASET.RENI_HDR.IS_HDR = True # If True, will use HDR transforms

_CN.DATASET.RENI_LDR = CN()
_CN.DATASET.RENI_LDR.PATH = "data/RENI_LDR"
_CN.DATASET.RENI_LDR.TRANSFORMS = [] # resize and totensor applied automatically
_CN.DATASET.RENI_LDR.IS_HDR = False # If True, will use HDR transforms

_CN.DATASET.CUSTOM = CN() # For other custom datasets
_CN.DATASET.CUSTOM.PATH = "data/custom"
_CN.DATASET.CUSTOM.TRANSFORMS = [] # resize and totensor applied automatically
_CN.DATASET.CUSTOM.IS_HDR = False # If True, will use HDR transforms

##############  ↓  TRAINER  ↓  ##############
_CN.TRAINER = CN()
_CN.TRAINER.LOGGER_TYPE = "tensorboard" # tensorboard | wandb
_CN.TRAINER.SEED = 42 # Random seed
_CN.TRAINER.MIXED_PRECISION = False # If True, will use mixed precision training will not work with Inverse Task
_CN.TRAINER.MAX_RUNTIME = 24 # hours

_CN.TRAINER.CHKPTS = CN()
_CN.TRAINER.CHKPTS.SAVE = True
_CN.TRAINER.CHKPTS.SAVE_DIR = "checkpoints"
_CN.TRAINER.CHKPTS.EVERY_N_EPOCHS = 10
_CN.TRAINER.CHKPTS.LOAD_PATH = None # Path of lightning checkpoint to load, if RENI.TASKS contains FIT_DECODER, decoder weights will be loaded but optimisation will start from scratch

_CN.TRAINER.LOGGER = CN()
_CN.TRAINER.LOGGER.LOG_IMAGES = True
_CN.TRAINER.LOGGER.NUMBER_OF_IMAGES = 10 # number of example images to log if 'IMAGES_TO_SHOW' is 'noise' or 'random'
_CN.TRAINER.LOGGER.IMAGES_TO_SHOW = "noise" # noise (radom latent codes) | random (random dataset idx) | [0, 1, 2, 3, 5, 6, 7, 8, 9, 10] (specific idx)
_CN.TRAINER.LOGGER.EPOCHS_BETWEEN_EXAMPLES = 1 # How often to log example images

_CN.TRAINER.LOGGER.WANDB = CN()
_CN.TRAINER.LOGGER.WANDB.NAME = "RENI"
_CN.TRAINER.LOGGER.WANDB.PROJECT = "RENI"
_CN.TRAINER.LOGGER.WANDB.SAVE_DIR = "wandb"
_CN.TRAINER.LOGGER.WANDB.OFFLINE = False
_CN.TRAINER.LOGGER.WANDB.LOG_MODEL = True

_CN.TRAINER.LOGGER.TB = CN()
_CN.TRAINER.LOGGER.TB.SAVE_DIR = "models" # where to save the tensorboard logs
_CN.TRAINER.LOGGER.TB.NAME = "auto" # Name of the experiment or 'auto' for auto naming
_CN.TRAINER.LOGGER.TB.LOG_GRAPH = True


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _CN.clone()