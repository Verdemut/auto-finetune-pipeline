class Preprocessor:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def process(self, dataset):
        self.logger.info("Preprocessing data...")
        return dataset