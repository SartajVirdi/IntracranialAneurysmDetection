
#!/usr/bin/env python
# coding: utf-8

# # RSNA Intracranial Aneurysm Detection - Phase 1 Enhanced Model 2

# ## 1. Setup and Imports

import os
import sys
import gc
import json
import shutil
import warnings
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Union
from contextlib import contextmanager
from IPython.display import display

warnings.filterwarnings('ignore')

# Data handling
import numpy as np
import polars as pl
import pandas as pd

# Medical imaging
import pydicom
import cv2

# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Competition API
import kaggle_evaluation.rsna_inference_server

# Set device with better error handling
def setup_device():
    """Setup and validate device configuration"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("Using CPU - inference will be slower")
    return device

device = setup_device()

# ## 2. Enhanced Configuration

# Competition constants
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery', 
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Model selection
SELECTED_MODEL = 'tf_efficientnetv2_s'

# Enhanced model paths
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class EnhancedInferenceConfig:
    """Phase 1 Enhanced configuration with accuracy improvements"""
    
    def __init__(self):
        # Model selection
        self.model_selection = SELECTED_MODEL
        self.use_ensemble = (SELECTED_MODEL == 'ensemble')
        
        # Model settings
        self.image_size = 512
        self.num_slices = 32
        self.use_windowing = True
        
        # Phase 1 Enhancement: Enhanced TTA
        self.use_tta = True
        self.tta_transforms = 8  # Increased from 4
        self.tta_weights = [0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05, 0.05]  # Different weights for transforms
        
        # Phase 1 Enhancement: Enhanced multichannel processing
        self.use_enhanced_multichannel = True
        self.use_percentile_projections = True
        self.use_modality_specific_processing = True
        
        # Phase 1 Enhancement: Medical constraint post-processing
        self.apply_medical_constraints = True
        self.symmetry_constraint_weight = 0.1
        self.aneurysm_consistency_weight = 0.2
        
        # Inference settings
        self.batch_size = 1
        self.use_amp = torch.cuda.is_available()
        
        # Memory management
        self.enable_memory_cleanup = True
        self.cleanup_frequency = 10
        
        # Error handling
        self.max_retries = 2
        self.fallback_enabled = True
        
        # Enhanced windowing parameters with more modalities
        self.windowing_params = {
            'CT': (40, 80),
            'CTA': (50, 350), 
            'MRA': (600, 1200),
            'MRI': (40, 80),
            'TOF': (500, 1000),  # Time of Flight
            'PC': (300, 600),    # Phase Contrast
            'FLAIR': (100, 300), # FLAIR sequences
            'default': (40, 80)
        }
        
        self.validate()
    
    def validate(self):
        """Validate configuration parameters"""
        assert self.image_size > 0, "Image size must be positive"
        assert self.num_slices > 0, "Number of slices must be positive"
        assert self.batch_size > 0, "Batch size must be positive"
        assert len(self.tta_weights) == self.tta_transforms, "TTA weights must match number of transforms"
        assert abs(sum(self.tta_weights) - 1.0) < 1e-6, "TTA weights must sum to 1"
        
        if self.model_selection not in MODEL_PATHS and self.model_selection != 'ensemble':
            raise ValueError(f"Unknown model: {self.model_selection}")

CFG = EnhancedInferenceConfig()

# ## 3. Enhanced Model Architecture (keeping existing robustness)

class MultiBackboneModel(nn.Module):
    """Enhanced model maintaining compatibility with existing checkpoints"""
    
    def __init__(self, model_name: str, num_classes: int = 14, pretrained: bool = True, 
                 drop_rate: float = 0.3, drop_path_rate: float = 0.2):
        super().__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        try:
            self._create_backbone(model_name, pretrained, drop_rate, drop_path_rate)
            self._determine_feature_dimensions()
            self._create_classifier(drop_rate)
        except Exception as e:
            print(f"Error initializing model {model_name}: {e}")
            raise
    
    def _create_backbone(self, model_name: str, pretrained: bool, drop_rate: float, drop_path_rate: float):
        """Create the backbone model"""
        backbone_kwargs = {
            'pretrained': pretrained,
            'in_chans': 3,
            'drop_rate': drop_rate,
            'num_classes': 0,
            'global_pool': ''
        }
        
        if 'swin' in model_name:
            backbone_kwargs.update({
                'drop_path_rate': drop_path_rate,
                'img_size': CFG.image_size,
            })
        elif 'convnext' in model_name:
            backbone_kwargs['drop_path_rate'] = drop_path_rate
            
        self.backbone = timm.create_model(model_name, **backbone_kwargs)
    
    def _determine_feature_dimensions(self):
        """Determine feature dimensions and pooling requirements"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            self.feature_shape = features.shape
            
            if len(features.shape) == 4:
                self.num_features = features.shape[1]
                self.needs_pool = True
                self.needs_seq_pool = False
            elif len(features.shape) == 3:
                self.num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                self.num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {self.model_name}: {self.num_features} features, shape: {self.feature_shape}")
        
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
    
    def _create_classifier(self, drop_rate: float):
        """Create metadata processing and classification layers"""
        # Keep existing metadata processing (2 features: age, sex)
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Keep existing classifier structure for checkpoint compatibility
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, self.num_classes)
        )
    
    def forward(self, image: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        """Forward pass with enhanced error handling"""
        try:
            img_features = self.backbone(image)
            img_features = self._pool_features(img_features)
            meta_features = self.meta_fc(meta)
            combined = torch.cat([img_features, meta_features], dim=1)
            output = self.classifier(combined)
            return output
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            return torch.zeros(image.size(0), self.num_classes, device=image.device)
    
    def _pool_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply appropriate pooling based on feature dimensions"""
        if self.needs_pool:
            features = self.global_pool(features)
            features = features.flatten(1)
        elif self.needs_seq_pool:
            features = features.mean(dim=1)
        elif len(features.shape) == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        elif len(features.shape) == 3:
            features = features.mean(dim=1)
        
        return features

# ## 4. Phase 1 Enhancement: Enhanced Multi-Channel Processing

@contextmanager
def dicom_error_handler(filepath: str):
    """Context manager for DICOM processing errors"""
    try:
        yield
    except Exception as e:
        print(f"DICOM processing error for {filepath}: {str(e)[:100]}...")
        raise

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing with enhanced validation"""
    if window_width <= 0:
        window_width = 1.0
    
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    
    if img_max <= img_min:
        img_max = img_min + 1
    
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    """Get appropriate windowing for different modalities"""
    return CFG.windowing_params.get(modality, CFG.windowing_params['default'])

def create_enhanced_multichannel_input(volume: np.ndarray, modality: str) -> np.ndarray:
    """
    Phase 1 Enhancement: Create enhanced multi-channel input with modality-specific processing
    """
    try:
        if volume.size == 0:
            empty_slice = np.zeros((CFG.image_size, CFG.image_size), dtype=np.uint8)
            return np.stack([empty_slice, empty_slice, empty_slice], axis=-1)
        
        # Modality-specific processing
        if modality in ['CTA', 'MRA', 'TOF'] and CFG.use_modality_specific_processing:
            # For angiographic sequences - focus on vessel enhancement
            
            # Channel 1: Maximum Intensity Projection (shows bright vessels)
            mip = np.max(volume, axis=0)
            
            # Channel 2: Minimum Intensity Projection (shows contrast differences)
            min_proj = np.min(volume, axis=0)
            
            # Channel 3: Vessel-enhanced projection (95th - 5th percentile)
            if CFG.use_percentile_projections:
                p95 = np.percentile(volume, 95, axis=0)
                p5 = np.percentile(volume, 5, axis=0)
                vessel_proj = (p95 - p5).astype(np.float32)
                
                # Normalize vessel projection
                if vessel_proj.max() > vessel_proj.min():
                    vessel_proj = ((vessel_proj - vessel_proj.min()) / 
                                 (vessel_proj.max() - vessel_proj.min()) * 255).astype(np.uint8)
                else:
                    vessel_proj = np.full_like(vessel_proj, 128, dtype=np.uint8)
            else:
                # Fallback to standard deviation
                vessel_proj = np.std(volume.astype(np.float32), axis=0).astype(np.uint8)
            
            image = np.stack([mip, min_proj, vessel_proj], axis=-1)
            
        else:
            # For CT/MRI - use enhanced standard processing
            
            # Channel 1: Middle slice (anatomical reference)
            middle_idx = volume.shape[0] // 2
            middle_slice = volume[middle_idx]
            
            # Channel 2: Maximum Intensity Projection
            mip = np.max(volume, axis=0)
            
            # Channel 3: Enhanced texture projection
            if CFG.use_percentile_projections:
                # Use 75th percentile projection for better texture representation
                p75 = np.percentile(volume, 75, axis=0).astype(np.float32)
                
                # Normalize
                if p75.max() > p75.min():
                    texture_proj = ((p75 - p75.min()) / 
                                  (p75.max() - p75.min()) * 255).astype(np.uint8)
                else:
                    texture_proj = np.full_like(p75, 128, dtype=np.uint8)
            else:
                # Standard deviation projection
                std_proj = np.std(volume.astype(np.float32), axis=0)
                if std_proj.max() > std_proj.min():
                    texture_proj = ((std_proj - std_proj.min()) / 
                                  (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
                else:
                    texture_proj = np.full_like(std_proj, 128, dtype=np.uint8)
            
            image = np.stack([middle_slice, mip, texture_proj], axis=-1)
        
        # Validate image shape
        if image.shape != (CFG.image_size, CFG.image_size, 3):
            raise ValueError(f"Invalid image shape: {image.shape}")
        
        return image
        
    except Exception as e:
        print(f"Error creating enhanced multichannel input: {e}")
        # Return default image
        default_slice = np.full((CFG.image_size, CFG.image_size), 128, dtype=np.uint8)
        return np.stack([default_slice, default_slice, default_slice], axis=-1)

# Continue with existing DICOM processing functions but use enhanced multichannel
def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image with robust statistics"""
    if img.size == 0:
        return img
    
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img = np.full_like(img, 128, dtype=np.uint8)
    
    return img

def extract_metadata(ds: pydicom.Dataset) -> Dict:
    """Extract metadata from DICOM with robust error handling"""
    metadata = {}
    
    # Modality with enhanced detection
    metadata['modality'] = getattr(ds, 'Modality', 'CT')
    
    # Enhanced modality detection from sequence names
    try:
        sequence_name = getattr(ds, 'SequenceName', '').upper()
        series_description = getattr(ds, 'SeriesDescription', '').upper()
        
        if 'TOF' in sequence_name or 'TOF' in series_description:
            metadata['modality'] = 'TOF'
        elif 'PC' in sequence_name or 'PHASE' in series_description:
            metadata['modality'] = 'PC'
        elif 'FLAIR' in sequence_name or 'FLAIR' in series_description:
            metadata['modality'] = 'FLAIR'
    except:
        pass
    
    # Age extraction
    try:
        age_str = getattr(ds, 'PatientAge', '050Y')
        age_digits = ''.join(filter(str.isdigit, str(age_str)[:3]))
        age = int(age_digits) if age_digits else 50
        metadata['age'] = max(0, min(age, 120))
    except:
        metadata['age'] = 50
    
    # Sex extraction
    try:
        sex = getattr(ds, 'PatientSex', 'M')
        metadata['sex'] = 1 if str(sex).upper().startswith('M') else 0
    except:
        metadata['sex'] = 0
    
    return metadata

def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    """Enhanced DICOM series processing with Phase 1 improvements"""
    series_path = Path(series_path)
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            filepath = os.path.join(root, file)
            if file.lower().endswith(('.dcm', '.dicom')) or _is_dicom_file(filepath):
                all_filepaths.append(filepath)
    
    all_filepaths.sort()
    
    if len(all_filepaths) == 0:
        print(f"Warning: No DICOM files found in {series_path}")
        return _get_default_volume_and_metadata()
    
    # Process DICOM files
    slices = []
    metadata = {}
    processing_errors = 0
    
    for i, filepath in enumerate(all_filepaths):
        try:
            with dicom_error_handler(filepath):
                ds = pydicom.dcmread(filepath, force=True)
                img = ds.pixel_array.astype(np.float32)
                
                # Handle multi-frame or color images
                img = _process_pixel_array(img)
                
                # Extract metadata from first file
                if i == 0 and not metadata:
                    metadata = extract_metadata(ds)
                
                # Apply rescale
                img = _apply_rescale(img, ds)
                
                # Apply enhanced windowing
                if CFG.use_windowing:
                    window_center, window_width = get_windowing_params(metadata.get('modality', 'CT'))
                    img = apply_dicom_windowing(img, window_center, window_width)
                else:
                    img = normalize_image(img)
                
                # Resize
                try:
                    img = cv2.resize(img, (CFG.image_size, CFG.image_size), interpolation=cv2.INTER_LINEAR)
                except cv2.error:
                    img = np.zeros((CFG.image_size, CFG.image_size), dtype=np.uint8)
                
                slices.append(img)
                
        except Exception as e:
            processing_errors += 1
            if processing_errors > len(all_filepaths) * 0.5:
                print(f"Too many processing errors, using default volume")
                return _get_default_volume_and_metadata()
            continue
    
    # Use default metadata if extraction failed
    if not metadata:
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
    
    # Create volume
    volume = _create_volume_from_slices(slices)
    
    return volume, metadata

# Helper functions (keeping existing implementations)
def _is_dicom_file(filepath: str) -> bool:
    """Check if file is DICOM by reading header"""
    try:
        with open(filepath, 'rb') as f:
            f.seek(128)
            dicm = f.read(4)
            return dicm == b'DICM'
    except:
        return False

def _process_pixel_array(img: np.ndarray) -> np.ndarray:
    """Process pixel array to handle different formats"""
    if img.ndim == 3:
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            img = img[img.shape[0] // 2]
    elif img.ndim > 3:
        img = img.reshape(img.shape[-2], img.shape[-1])
    
    return img

def _apply_rescale(img: np.ndarray, ds: pydicom.Dataset) -> np.ndarray:
    """Apply DICOM rescale with validation"""
    try:
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            slope = float(ds.RescaleSlope)
            intercept = float(ds.RescaleIntercept)
            if slope != 0:
                img = img * slope + intercept
    except:
        pass
    
    return img

def _create_volume_from_slices(slices: List[np.ndarray]) -> np.ndarray:
    """Create volume from slices with enhanced handling"""
    if len(slices) == 0:
        return np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    
    volume = np.array(slices)
    
    if len(slices) > CFG.num_slices:
        indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
        volume = volume[indices]
    elif len(slices) < CFG.num_slices:
        pad_size = CFG.num_slices - len(slices)
        if len(slices) == 1:
            volume = np.repeat(volume, CFG.num_slices, axis=0)
        else:
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    return volume

def _get_default_volume_and_metadata() -> Tuple[np.ndarray, Dict]:
    """Return default volume and metadata for error cases"""
    volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
    return volume, metadata

# ## 5. Phase 1 Enhancement: Advanced Test-Time Augmentation

def get_inference_transform() -> A.Compose:
    """Get robust inference transformation"""
    return A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

def get_enhanced_tta_transforms() -> List[A.Compose]:
    """
    Phase 1 Enhancement: Advanced TTA transforms optimized for medical imaging
    """
    base_norm = [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ]
    
    transforms = [
        # 1. Original (highest weight)
        A.Compose(base_norm),
        
        # 2. Horizontal flip (common in medical imaging)
        A.Compose([
            A.HorizontalFlip(p=1.0)
        ] + base_norm),
        
        # 3. Vertical flip
        A.Compose([
            A.VerticalFlip(p=1.0)
        ] + base_norm),
        
        # 4. Small rotation (medical images can have slight rotation)
        A.Compose([
            A.Rotate(limit=10, border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0)
        ] + base_norm),
        
        # 5. Brightness/Contrast (different scanner settings)
        A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=0.1, 
                contrast_limit=0.1, 
                p=1.0
            )
        ] + base_norm),
        
        # 6. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0)
        ] + base_norm),
        
        # 7. Gaussian blur (slight smoothing)
        A.Compose([
            A.GaussianBlur(blur_limit=(1, 3), sigma_limit=0, p=1.0)
        ] + base_norm),
        
        # 8. Sharpening (edge enhancement)
        A.Compose([
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.8, 1.2), p=1.0)
        ] + base_norm),
    ]
    
    return transforms

# ## 6. Enhanced Model Loading (keeping existing robustness)

# Global variables
MODELS: Dict[str, nn.Module] = {}
TRANSFORM: Optional[A.Compose] = None
TTA_TRANSFORMS: Optional[List[A.Compose]] = None
PREDICTION_COUNT = 0

def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """Load a single model with enhanced error handling"""
    print(f"Loading {model_name} from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        model_config = checkpoint.get('model_config', {})
        training_config = checkpoint.get('training_config', {})
        
        if 'image_size' in training_config:
            CFG.image_size = training_config['image_size']
            print(f"Updated image size to {CFG.image_size}")
        
        model = MultiBackboneModel(
            model_name=model_name,
            num_classes=training_config.get('num_classes', 14),
            pretrained=False,
            drop_rate=0.0,
            drop_path_rate=0.0
        )
        
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        except RuntimeError as e:
            print(f"Warning: Strict loading failed, trying non-strict: {e}")
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        model = model.to(device)
        model.eval()
        
        _validate_model(model)
        
        best_score = checkpoint.get('best_score', 'N/A')
        print(f"✓ Loaded {model_name} successfully (best score: {best_score})")
        
        return model
        
    except Exception as e:
        print(f"✗ Failed to load {model_name}: {e}")
        raise

def _validate_model(model: nn.Module):
    """Validate that model works correctly"""
    try:
        with torch.no_grad():
            dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size, device=device)
            dummy_meta = torch.randn(1, 2, device=device)
            output = model(dummy_image, dummy_meta)
            
            if output.shape != (1, len(LABEL_COLS)):
                raise ValueError(f"Model output shape {output.shape} != expected {(1, len(LABEL_COLS))}")
                
    except Exception as e:
        raise RuntimeError(f"Model validation failed: {e}")

def load_models():
    """Load models based on configuration"""
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    print("=" * 50)
    print("Loading Phase 1 Enhanced Model...")
    
    successful_loads = 0
    
    try:
        if CFG.use_ensemble:
            for model_name, model_path in MODEL_PATHS.items():
                try:
                    MODELS[model_name] = load_single_model(model_name, model_path)
                    successful_loads += 1
                except Exception as e:
                    print(f"⚠ Warning: Could not load {model_name}: {e}")
                    if CFG.fallback_enabled:
                        continue
                    else:
                        raise
        else:
            if CFG.model_selection in MODEL_PATHS:
                model_path = MODEL_PATHS[CFG.model_selection]
                MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
                successful_loads = 1
            else:
                raise ValueError(f"Unknown model: {CFG.model_selection}")
        
        if successful_loads == 0:
            raise RuntimeError("No models loaded successfully!")
        
        # Initialize transforms
        TRANSFORM = get_inference_transform()
        if CFG.use_tta:
            TTA_TRANSFORMS = get_enhanced_tta_transforms()
        
        print(f"✓ Models loaded successfully: {list(MODELS.keys())}")
        print(f"✓ Enhanced TTA transforms: {len(TTA_TRANSFORMS) if TTA_TRANSFORMS else 0}")
        
        # Comprehensive model warmup
        _warmup_models()
        
        print("✓ Ready for Phase 1 enhanced inference!")
        print("=" * 50)
        
    except Exception as e:
        print(f"✗ Critical error loading models: {e}")
        if not CFG.fallback_enabled:
            raise
        print("Continuing with fallback mode...")

def _warmup_models():
    """Comprehensive model warmup"""
    print("Warming up models...")
    
    try:
        dummy_image = torch.randn(CFG.batch_size, 3, CFG.image_size, CFG.image_size, device=device)
        dummy_meta = torch.randn(CFG.batch_size, 2, device=device)
        
        with torch.no_grad():
            for model_name, model in MODELS.items():
                for _ in range(3):
                    if CFG.use_amp:
                        with autocast():
                            _ = model(dummy_image, dummy_meta)
                    else:
                        _ = model(dummy_image, dummy_meta)
                
                print(f"✓ Warmed up {model_name}")
        
        del dummy_image, dummy_meta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Warning: Model warmup failed: {e}")

# ## 7. Phase 1 Enhancement: Medical Constraint Post-Processing

def apply_medical_constraints(predictions: np.ndarray) -> np.ndarray:
    """
    Phase 1 Enhancement: Apply anatomical and medical knowledge constraints
    """
    try:
        if not CFG.apply_medical_constraints:
            return predictions
        
        predictions = predictions.copy()  # Don't modify original
        
        # Constraint 1: Aneurysm consistency
        # If "Aneurysm Present" is very low, reduce all specific location probabilities
        aneurysm_present = predictions[-1]
        
        if aneurysm_present < 0.05:  # Very low aneurysm probability
            # Reduce specific locations more aggressively
            predictions[:-1] *= (0.3 + 0.7 * aneurysm_present / 0.05)
        elif aneurysm_present < 0.15:  # Low aneurysm probability  
            # Moderate reduction
            predictions[:-1] *= (0.6 + 0.4 * aneurysm_present / 0.15)
        
        # Constraint 2: Bilateral symmetry consideration
        # Left and right anatomical structures should have some correlation
        if CFG.symmetry_constraint_weight > 0:
            # Carotid arteries (indices 0,1,2,3)
            left_carotid_avg = (predictions[0] + predictions[2]) / 2  # Left infra + supra
            right_carotid_avg = (predictions[1] + predictions[3]) / 2  # Right infra + supra
            
            # Middle cerebral arteries (indices 4,5)
            left_mca = predictions[4]
            right_mca = predictions[5]
            
            # Anterior cerebral arteries (indices 7,8)
            left_aca = predictions[7]
            right_aca = predictions[8]
            
            # Posterior communicating arteries (indices 9,10)
            left_pcom = predictions[9]
            right_pcom = predictions[10]
            
            # Apply soft symmetry constraints
            pairs = [
                (left_carotid_avg, right_carotid_avg, [0, 2], [1, 3]),
                (left_mca, right_mca, [4], [5]),
                (left_aca, right_aca, [7], [8]),
                (left_pcom, right_pcom, [9], [10])
            ]
            
            for left_val, right_val, left_indices, right_indices in pairs:
                if abs(left_val - right_val) > 0.3:  # High asymmetry
                    # Slightly reduce extreme asymmetry
                    avg_val = (left_val + right_val) / 2
                    adjustment = CFG.symmetry_constraint_weight * (avg_val - left_val)
                    
                    for idx in left_indices:
                        predictions[idx] = max(0.001, min(0.999, predictions[idx] + adjustment))
                    
                    adjustment = CFG.symmetry_constraint_weight * (avg_val - right_val)
                    for idx in right_indices:
                        predictions[idx] = max(0.001, min(0.999, predictions[idx] + adjustment))
        
        # Constraint 3: Anatomical region consistency
        # If multiple locations in same region are predicted, slightly boost "Aneurysm Present"
        anterior_locations = predictions[[0, 1, 2, 3, 4, 5, 6, 7, 8]]  # Anterior circulation
        posterior_locations = predictions[[9, 10, 11, 12]]  # Posterior circulation
        
        max_anterior = np.max(anterior_locations)
        max_posterior = np.max(posterior_locations)
        
        if max_anterior > 0.3 or max_posterior > 0.3:
            # Boost aneurysm present slightly
            boost_factor = min(0.1, CFG.aneurysm_consistency_weight * max(max_anterior, max_posterior))
            predictions[-1] = min(0.999, predictions[-1] + boost_factor)
        
        # Final validation
        predictions = np.clip(predictions, 0.001, 0.999)
        
        return predictions
        
    except Exception as e:
        print(f"Error applying medical constraints: {e}")
        return predictions

# ## 8. Enhanced Prediction Functions

def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """Make prediction with enhanced TTA and error handling"""
    try:
        predictions = []
        
        if CFG.use_tta and TTA_TRANSFORMS:
            # Phase 1 Enhancement: Weighted TTA with 8 transforms
            for i, (transform, weight) in enumerate(zip(TTA_TRANSFORMS[:CFG.tta_transforms], CFG.tta_weights)):
                try:
                    aug_image = transform(image=image)['image']
                    aug_image = aug_image.unsqueeze(0).to(device, non_blocking=True)
                    
                    with torch.no_grad():
                        if CFG.use_amp:
                            with autocast():
                                output = model(aug_image, meta_tensor)
                        else:
                            output = model(aug_image, meta_tensor)
                        
                        pred = torch.sigmoid(output).cpu().numpy().squeeze()
                        predictions.append((pred, weight))
                        
                except Exception as e:
                    print(f"Warning: TTA transform {i} failed: {e}")
                    continue
            
            if predictions:
                # Weighted average of TTA predictions
                weighted_preds = np.array([pred for pred, weight in predictions])
                weights = np.array([weight for pred, weight in predictions])
                weights = weights / weights.sum()  # Renormalize weights
                
                final_pred = np.average(weighted_preds, weights=weights, axis=0)
                return final_pred
            else:
                print("All TTA transforms failed, using single prediction")
        
        # Single prediction fallback
        image_tensor = TRANSFORM(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(device, non_blocking=True)
        
        with torch.no_grad():
            if CFG.use_amp:
                with autocast():
                    output = model(image_tensor, meta_tensor)
            else:
                output = model(image_tensor, meta_tensor)
            
            return torch.sigmoid(output).cpu().numpy().squeeze()
            
    except Exception as e:
        print(f"Error in single model prediction: {e}")
        return np.full(len(LABEL_COLS), 0.1)

def _manage_memory():
    """Enhanced memory management"""
    global PREDICTION_COUNT
    
    PREDICTION_COUNT += 1
    
    if CFG.enable_memory_cleanup and PREDICTION_COUNT % CFG.cleanup_frequency == 0:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
        print(f"Memory cleanup performed after {PREDICTION_COUNT} predictions")

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Enhanced prediction logic with Phase 1 improvements"""
    global MODELS
    
    if not MODELS:
        load_models()
        if not MODELS and not CFG.fallback_enabled:
            raise RuntimeError("No models available and fallback disabled")
    
    series_id = os.path.basename(series_path)
    
    try:
        # Process DICOM series with retry logic
        volume, metadata = None, None
        for attempt in range(CFG.max_retries):
            try:
                volume, metadata = process_dicom_series(series_path)
                break
            except Exception as e:
                print(f"DICOM processing attempt {attempt + 1} failed: {e}")
                if attempt == CFG.max_retries - 1:
                    raise
                
        if volume is None or metadata is None:
            raise ValueError("Failed to process DICOM series")
        
        # Phase 1 Enhancement: Create enhanced multi-channel input
        if CFG.use_enhanced_multichannel:
            image = create_enhanced_multichannel_input(volume, metadata.get('modality', 'CT'))
        else:
            # Fallback to original method
            middle_slice = volume[CFG.num_slices // 2]
            mip = np.max(volume, axis=0)
            std_proj = np.std(volume.astype(np.float32), axis=0)
            std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min() + 1e-7) * 255).astype(np.uint8)
            image = np.stack([middle_slice, mip, std_proj], axis=-1)
        
        # Prepare metadata tensor
        meta_tensor = _prepare_metadata_tensor(metadata)
        
        # Make predictions
        if CFG.use_ensemble and len(MODELS) > 1:
            # Use ensemble if multiple models available
            all_predictions = []
            for model_name, model in MODELS.items():
                try:
                    pred = predict_single_model(model, image, meta_tensor)
                    all_predictions.append(pred)
                except Exception as e:
                    print(f"Warning: Model {model_name} prediction failed: {e}")
                    continue
            
            if all_predictions:
                final_pred = np.mean(all_predictions, axis=0)
            else:
                raise ValueError("All ensemble models failed")
        else:
            # Use single model
            model_name = CFG.model_selection if CFG.model_selection in MODELS else list(MODELS.keys())[0]
            model = MODELS[model_name]
            final_pred = predict_single_model(model, image, meta_tensor)
        
        if final_pred is None:
            raise ValueError("Prediction failed")
        
        # Phase 1 Enhancement: Apply medical constraints
        final_pred = apply_medical_constraints(final_pred)
        
        # Validate predictions
        final_pred = _validate_predictions(final_pred)
        
        # Create output dataframe
        predictions_df = pl.DataFrame(
            data=[final_pred.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        # Perform memory management
        _manage_memory()
        
        return predictions_df
        
    except Exception as e:
        print(f"Error in prediction for {series_id}: {e}")
        if CFG.fallback_enabled:
            return _create_fallback_predictions()
        else:
            raise

def _prepare_metadata_tensor(metadata: Dict) -> torch.Tensor:
    """Prepare metadata tensor with validation"""
    try:
        age_normalized = np.clip(metadata.get('age', 50) / 100.0, 0.0, 1.2)
        sex = int(metadata.get('sex', 0))
        sex = np.clip(sex, 0, 1)
        
        meta_tensor = torch.tensor(
            [[age_normalized, sex]], 
            dtype=torch.float32, 
            device=device
        )
        
        return meta_tensor
        
    except Exception as e:
        print(f"Error preparing metadata: {e}")
        return torch.tensor([[0.5, 0.0]], dtype=torch.float32, device=device)

def _validate_predictions(predictions: np.ndarray) -> np.ndarray:
    """Validate and clean predictions"""
    try:
        if predictions.shape != (len(LABEL_COLS),):
            print(f"Warning: Prediction shape {predictions.shape} != expected {(len(LABEL_COLS),)}")
            predictions = np.resize(predictions, len(LABEL_COLS))
        
        if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
            print("Warning: Invalid values in predictions, clipping")
            predictions = np.nan_to_num(predictions, nan=0.1, posinf=0.9, neginf=0.0)
        
        predictions = np.clip(predictions, 0.001, 0.999)
        
        return predictions
        
    except Exception as e:
        print(f"Error validating predictions: {e}")
        return np.full(len(LABEL_COLS), 0.1)

def _create_fallback_predictions() -> pl.DataFrame:
    """Create fallback predictions"""
    fallback_values = [0.05] * (len(LABEL_COLS) - 1) + [0.1]
    
    return pl.DataFrame(
        data=[fallback_values],
        schema=LABEL_COLS,
        orient='row'
    )

# ## 9. Error Handling and Fallback (keeping existing robustness)

def predict_fallback(series_path: str) -> pl.DataFrame:
    """Enhanced fallback prediction function"""
    series_id = os.path.basename(series_path)
    print(f"Using fallback predictions for {series_id}")
    
    try:
        shared_dir = '/kaggle/shared'
        if os.path.exists(shared_dir):
            shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        return _create_fallback_predictions()
        
    except Exception as e:
        print(f"Error in fallback: {e}")
        fallback_values = [0.05] * len(LABEL_COLS)
        return pl.DataFrame(
            data=[fallback_values],
            schema=LABEL_COLS,
            orient='row'
        )

def predict(series_path: str) -> pl.DataFrame:
    """
    Enhanced top-level prediction function with Phase 1 improvements
    """
    start_time = None
    
    try:
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        if start_time:
            start_time.record()
        
        if not os.path.exists(series_path):
            raise FileNotFoundError(f"Series path does not exist: {series_path}")
        
        result = _predict_inner(series_path)
        
        if start_time and torch.cuda.is_available():
            end_time = torch.cuda.Event(enable_timing=True)
            end_time.record()
            torch.cuda.synchronize()
            elapsed = start_time.elapsed_time(end_time)
            print(f"Phase 1 enhanced prediction completed in {elapsed:.1f}ms")
        
        return result
        
    except FileNotFoundError as e:
        print(f"File error for {os.path.basename(series_path)}: {e}")
        return predict_fallback(series_path)
        
    except RuntimeError as e:
        print(f"Runtime error for {os.path.basename(series_path)}: {e}")
        return predict_fallback(series_path)
        
    except Exception as e:
        print(f"Unexpected error for {os.path.basename(series_path)}: {e}")
        if hasattr(e, '__traceback__'):
            print("Traceback:")
            traceback.print_exc()
        return predict_fallback(series_path)
        
    finally:
        try:
            shared_dir = '/kaggle/shared'
            if os.path.exists(shared_dir):
                shutil.rmtree(shared_dir, ignore_errors=True)
            os.makedirs(shared_dir, exist_ok=True)
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            
        except Exception as cleanup_error:
            print(f"Warning: Cleanup failed: {cleanup_error}")

# ## 10. Enhanced Main Execution

def main():
    """Main execution function with Phase 1 enhancements"""
    try:
        print("=" * 70)
        print("RSNA Intracranial Aneurysm Detection - Phase 1 Enhanced Model")
        print("=" * 70)
        
        # Print Phase 1 enhancements
        print("Phase 1 Enhancements Active:")
        print(f"✓ Enhanced Multi-Channel Processing: {CFG.use_enhanced_multichannel}")
        print(f"✓ Modality-Specific Processing: {CFG.use_modality_specific_processing}")
        print(f"✓ Percentile Projections: {CFG.use_percentile_projections}")
        print(f"✓ Advanced TTA (8 transforms): {CFG.use_tta}")
        print(f"✓ Medical Constraint Post-Processing: {CFG.apply_medical_constraints}")
        print("-" * 70)
        
        # Print configuration
        print(f"Selected model: {CFG.model_selection}")
        print(f"Use ensemble: {CFG.use_ensemble}")
        print(f"Image size: {CFG.image_size}")
        print(f"Device: {device}")
        print("-" * 70)
        
        # Load models
        load_models()
        
        # Initialize the inference server
        print("Initializing inference server...")
        inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
        
        # Check environment and run appropriate mode
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Running in competition mode...")
            inference_server.serve()
        else:
            print("Running in local development mode...")
            inference_server.run_local_gateway()
            
            # Display results
            try:
                submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
                print("\nPhase 1 Enhanced Submission Summary:")
                print(f"Shape: {submission_df.shape}")
                
                # Enhanced statistics
                numeric_cols = [col for col in submission_df.columns if col != ID_COL]
                if numeric_cols:
                    print("\nPrediction Statistics (Phase 1 Enhanced):")
                    for col in numeric_cols:
                        mean_pred = submission_df[col].mean()
                        std_pred = submission_df[col].std() if len(submission_df) > 1 else 0
                        print(f"{col}: {mean_pred:.4f} (±{std_pred:.4f})")
                
                display(submission_df.head())
                
            except Exception as e:
                print(f"Could not display results: {e}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        traceback.print_exc()
    finally:
        print("Phase 1 enhanced inference complete!")

# Run main execution
if __name__ == "__main__":
    main()
else:
    # If imported as module, still load models
    load_models()
