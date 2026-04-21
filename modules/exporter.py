import torch
from pathlib import Path

class Exporter:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def export(self, model, metrics):
        self.logger.info("Exporting model...")
        
        output_path = Path(self.config['output_path'])
        
        if self.config['export']['save_full_model']:
            model_path = output_path / "final_model"
            model_path.mkdir(parents=True, exist_ok=True)
            
            if 'unet' in model:
                model['unet'].save_pretrained(model_path)
                self.logger.info(f"Model saved to: {model_path}")
            
            # Сохраняем конфиг
            import yaml
            with open(model_path / "config.yaml", 'w') as f:
                yaml.dump(self.config, f)
        
        self.logger.info("Export completed")