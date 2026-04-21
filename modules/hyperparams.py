# modules/hyperparams.py - Без эмодзи
class HyperparamOptimizer:
    def __init__(self, config, hardware, logger):
        self.config = config
        self.hardware = hardware
        self.logger = logger
    
    def optimize(self, dataset):
        self.logger.info("Optimizing hyperparameters...")
        
        dataset_size = len(dataset['train'])
        
        # 1. Выбор метода обучения
        if self.config['training']['method'] == "auto":
            method = self.hardware.get_optimal_method(dataset_size)
        else:
            method = self.config['training']['method']
        
        # 2. Batch size
        batch_size = self.config['training'].get('batch_size')
        if batch_size is None:
            batch_size = self.hardware.get_optimal_batch_size()
        else:
            batch_size = int(batch_size)
        
        # 3. Gradient accumulation
        grad_accum = self.config['training'].get('gradient_accumulation_steps')
        if grad_accum is None:
            grad_accum = self.hardware.get_optimal_gradient_accumulation(batch_size)
        else:
            grad_accum = int(grad_accum)
        
        # 4. Learning rate
        lr = self.config['training'].get('learning_rate')
        if lr is None:
            learning_rate = self.hardware.get_optimal_learning_rate(method)
        else:
            learning_rate = float(lr)
        
        # 5. Количество эпох
        num_epochs = self.config['training'].get('num_epochs')
        if num_epochs is None or num_epochs == "auto":
            num_epochs = self.hardware.get_optimal_num_epochs(dataset_size, method)
        else:
            num_epochs = int(num_epochs)
        
        # 6. FP16
        use_fp16 = self.config['model'].get('use_fp16')
        if use_fp16 is None:
            use_fp16 = self.hardware.get_optimal_fp16()
        
        # 7. xFormers
        use_xformers = self.config['model'].get('enable_xformers')
        if use_xformers is None:
            use_xformers = self.hardware.should_use_xformers()
        
        # Обновляем конфиг
        self.config['model']['use_fp16'] = use_fp16
        self.config['model']['enable_xformers'] = use_xformers
        
        hyperparams = {
            "method": method,
            "batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "use_fp16": use_fp16,
            "use_xformers": use_xformers
        }
        
        # Вывод подобранных параметров
        print(f"\nSELECTED HYPERPARAMETERS:")
        print(f"  Method: {method}")
        print(f"  Batch size: {batch_size}")
        print(f"  Gradient accumulation: {grad_accum}")
        print(f"  Effective batch size: {batch_size * grad_accum}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
        print(f"  FP16: {'Enabled' if use_fp16 else 'Disabled'}")
        print(f"  xFormers: {'Enabled' if use_xformers else 'Disabled'}")
        
        # Оценка времени
        self.hardware.print_training_estimate(dataset_size, num_epochs, batch_size)
        
        return hyperparams