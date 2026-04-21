# main.py - Без эмодзи
import os
import sys
import argparse
import yaml
import torch
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.hardware_check import HardwareChecker
from utils.logger import setup_logger
from modules.data_loader import DataLoader
from modules.preprocessor import Preprocessor
from modules.hyperparams import HyperparamOptimizer
from modules.trainer import Trainer
from modules.validator import Validator
from modules.exporter import Exporter


class AutoFinetunePipeline:
    def __init__(self, config_path):
        print("=" * 70)
        print("Auto-Finetune Pipeline v3.0")
        print("Professional Edition")
        print("=" * 70)
        
        # Загружаем конфиг
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Создаем папки
        for path in [self.config['output_path'], self.config['logs_path'], 
                     self.config['checkpoints_path']]:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Настраиваем логирование
        self.logger = setup_logger(self.config['logs_path'])
        self.logger.info("Auto-Finetune Pipeline v3.0 initialized")
        
        # Проверяем оборудование
        self.hardware = HardwareChecker(self.config)
        self.hardware.check()
        
        print(f"\nOutput directory: {self.config['output_path']}")
        print(f"Logs directory: {self.config['logs_path']}")
        
    def run(self):
        """Запуск пайплайна"""
        start_time = datetime.now()
        self.logger.info("Starting pipeline")
        
        try:
            # Шаг 1: Загрузка данных
            print("\n" + "=" * 70)
            print("STEP 1: DATA LOADING")
            print("=" * 70)
            data_loader = DataLoader(self.config, self.logger)
            dataset = data_loader.load()
            
            # Шаг 2: Предобработка
            print("\n" + "=" * 70)
            print("STEP 2: DATA PREPROCESSING")
            print("=" * 70)
            preprocessor = Preprocessor(self.config, self.logger)
            processed_data = preprocessor.process(dataset)
            
            # Шаг 3: Автоподбор гиперпараметров
            print("\n" + "=" * 70)
            print("STEP 3: HYPERPARAMETER OPTIMIZATION")
            print("=" * 70)
            hyper_optimizer = HyperparamOptimizer(self.config, self.hardware, self.logger)
            hyperparams = hyper_optimizer.optimize(processed_data)
            
            # Шаг 4: Обучение
            print("\n" + "=" * 70)
            print("STEP 4: MODEL TRAINING")
            print("=" * 70)
            trainer = Trainer(self.config, hyperparams, self.logger)
            trained_model = trainer.train(processed_data)
            
            # Шаг 5: Валидация
            print("\n" + "=" * 70)
            print("STEP 5: MODEL VALIDATION")
            print("=" * 70)
            validator = Validator(self.config, self.logger)
            metrics = validator.validate(trained_model, processed_data)
            
            # Шаг 6: Экспорт
            print("\n" + "=" * 70)
            print("STEP 6: MODEL EXPORT")
            print("=" * 70)
            exporter = Exporter(self.config, self.logger)
            exporter.export(trained_model, metrics)
            
            self.print_summary(metrics, start_time)
            
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
            print("\nTraining interrupted. Model saved in checkpoints.")
            
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            print(f"\nCritical error: {str(e)}")
            raise
    
    def print_summary(self, metrics, start_time):
        """Печать итогового отчета"""
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETED")
        print("=" * 70)
        print(f"Status: Success")
        print(f"Duration: {duration}")
        print(f"Model saved: {self.config['output_path']}")
        
        if metrics:
            print("\nMETRICS:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print("\nNEXT STEPS:")
        print(f"  1. Load model: pipe = StableDiffusionPipeline.from_pretrained('{self.config['output_path']}/final_model')")
        print(f"  2. Generate: pipe('your prompt').images[0].save('result.png')")


def main():
    parser = argparse.ArgumentParser(description="Auto-Finetune Pipeline v3.0")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--dataset', type=str,
                       help='Path to dataset (overrides config)')
    parser.add_argument('--output', type=str,
                       help='Output path (overrides config)')
    parser.add_argument('--resume', type=str,
                       help='Resume training from checkpoint')
    
    args = parser.parse_args()
    
    # Загружаем конфиг
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Переопределяем параметры
    if args.dataset:
        config['dataset_path'] = args.dataset
        print(f"Dataset: {args.dataset}")
    
    if args.output:
        config['output_path'] = args.output
        print(f"Output: {args.output}")
    
    if args.resume:
        config['resume_from'] = args.resume
        print(f"Resume from: {args.resume}")
    
    # Временный конфиг
    temp_config = f"temp_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(temp_config, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)
    
    try:
        pipeline = AutoFinetunePipeline(temp_config)
        pipeline.run()
    finally:
        if os.path.exists(temp_config):
            os.remove(temp_config)


if __name__ == "__main__":
    main()