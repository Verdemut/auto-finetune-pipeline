# modules/data_loader.py - Без эмодзи
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CustomDataset(Dataset):
    def __init__(self, image_paths, captions, config, augment=False):
        self.image_paths = image_paths
        self.captions = captions
        self.config = config
        self.image_size = config['dataset']['image_size']
        self.augment = augment
        
        transform_list = [
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ]
        
        if augment and config['dataset'].get('augmentation', {}).get('enabled', False):
            aug_config = config['dataset']['augmentation']
            if aug_config.get('horizontal_flip', False):
                transform_list.insert(0, transforms.RandomHorizontalFlip())
            if aug_config.get('random_rotation', 0):
                transform_list.insert(0, transforms.RandomRotation(aug_config['random_rotation']))
        
        self.transform = transforms.Compose(transform_list)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            image = Image.open(self.image_paths[idx]).convert('RGB')
            image_tensor = self.transform(image)
            caption = self.captions[idx]
            
            return {
                "pixel_values": image_tensor,
                "caption": caption,
                "path": str(self.image_paths[idx]),
                "idx": idx
            }
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            return self.__getitem__((idx + 1) % len(self.image_paths))


class DataLoader:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.dataset_path = Path(config['dataset_path'])
        self.image_size = config['dataset']['image_size']
        
    def load(self):
        self.logger.info(f"Loading dataset from {self.dataset_path}")
        
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {self.dataset_path}")
        
        image_paths = self._find_images()
        
        if not image_paths:
            raise FileNotFoundError(f"No images found in {self.dataset_path}")
        
        self.logger.info(f"Found images: {len(image_paths)}")
        
        captions = self._load_captions(image_paths)
        
        dataset = CustomDataset(image_paths, captions, self.config, augment=True)
        
        val_split = self.config['dataset'].get('validation_split', 0.1)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        self.logger.info(f"Train: {len(train_dataset)} images")
        self.logger.info(f"Validation: {len(val_dataset)} images")
        
        self._print_stats(captions)
        
        return {
            "train": train_dataset,
            "val": val_dataset,
            "all": dataset,
            "captions": captions
        }
    
    def _find_images(self):
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(list(self.dataset_path.glob(f"*{ext}")))
        
        if not image_paths:
            images_folder = self.dataset_path / "images"
            if images_folder.exists():
                for ext in image_extensions:
                    image_paths.extend(list(images_folder.glob(f"*{ext}")))
        
        return image_paths
    
    def _load_captions(self, image_paths):
        captions_file = self.dataset_path / "captions.csv"
        
        if not captions_file.exists():
            captions_file = self.dataset_path / "images" / "captions.csv"
        
        if captions_file.exists():
            try:
                df = pd.read_csv(captions_file, encoding='utf-8')
                
                if 'image' in df.columns and 'caption' in df.columns:
                    captions_dict = dict(zip(df['image'].astype(str), df['caption']))
                    captions = []
                    
                    for img_path in image_paths:
                        img_name = img_path.stem
                        caption = captions_dict.get(img_name, f"image_{img_name}")
                        captions.append(caption)
                    
                    self.logger.info(f"Loaded captions: {len(captions)}")
                    return captions
                else:
                    self.logger.warning(f"CSV must have 'image' and 'caption' columns. Found: {list(df.columns)}")
            except Exception as e:
                self.logger.error(f"Error reading CSV: {e}")
        
        self.logger.warning("captions.csv not found, using temporary captions")
        return [f"image_{i}" for i in range(len(image_paths))]
    
    def _print_stats(self, captions):
        caption_lengths = [len(c) for c in captions]
        
        print(f"\nDATASET STATISTICS:")
        print(f"  Total images: {len(captions)}")
        print(f"  Average caption length: {sum(caption_lengths) / len(caption_lengths):.0f} characters")
        print(f"  Min caption length: {min(caption_lengths)}")
        print(f"  Max caption length: {max(caption_lengths)}")