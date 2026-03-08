import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import shutil
import random
import zipfile

def setup_dataset(zip_path, extract_path='./original_dataset', base_dir='project_dataset', split_ratio=0.8, seed=42):
    """
    Estrae il dataset (se necessario) e crea gli split Train/Val/Test.
    - Test set originale rimane intatto.
    - Train set originale viene splittato 80/20 in Train e Val.
    - Validation set originale (16 img) viene aggiunto al nuovo Val set.
    """
    random.seed(seed)
    
    # 1. Estrazione Zip
    if not os.path.exists(extract_path) and os.path.exists(zip_path):
        print(f"Estrazione dataset da {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print("Dataset estratto!")
    elif not os.path.exists(extract_path):
        print(f"ATTENZIONE: Percorso {extract_path} non trovato e ZIP non presente.")
        return

    # Percorsi originali
    orig_train_norm = os.path.join(extract_path, 'chest_xray/train/NORMAL')
    orig_train_pneu = os.path.join(extract_path, 'chest_xray/train/PNEUMONIA')
    orig_test = os.path.join(extract_path, 'chest_xray/test')
    orig_val = os.path.join(extract_path, 'chest_xray/val')

    # Percorsi destinazione
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Se esiste già, evitiamo di ricrearlo
    if os.path.exists(base_dir):
        print(f"Dataset split già presente in {base_dir}")
        return train_dir, val_dir, test_dir, 0, 0

    print("Creazione split Train/Val/Test...")
    for cat in ['NORMAL', 'PNEUMONIA']:
        os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cat), exist_ok=True)

    # Copia Test Set intatto
    shutil.copytree(orig_test, test_dir)

    def split_and_copy(src, train_dst, val_dst, ratio=0.8):
        files = [f for f in os.listdir(src) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(files)
        sp = int(len(files) * ratio)
        for f in files[:sp]:
            shutil.copy(os.path.join(src, f), os.path.join(train_dst, f))
        for f in files[sp:]:
            shutil.copy(os.path.join(src, f), os.path.join(val_dst, f))
        return sp, len(files) - sp

    # Split Train Set originale
    n_train_n, n_val_n = split_and_copy(orig_train_norm,
        os.path.join(train_dir, 'NORMAL'), os.path.join(val_dir, 'NORMAL'), split_ratio)
    n_train_p, n_val_p = split_and_copy(orig_train_pneu,
        os.path.join(train_dir, 'PNEUMONIA'), os.path.join(val_dir, 'PNEUMONIA'), split_ratio)

    # Aggiungi le 16 immagini della validation originale al nuovo Val set
    n_orig_val = 0
    for cat in ['NORMAL', 'PNEUMONIA']:
        orig_val_cat = os.path.join(orig_val, cat)
        if os.path.exists(orig_val_cat):
            for f in os.listdir(orig_val_cat):
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    shutil.copy(os.path.join(orig_val_cat, f), os.path.join(val_dir, cat, f))
                    n_orig_val += 1
                    if cat == 'NORMAL': n_val_n += 1
                    else: n_val_p += 1

    print(f"Dataset generato con successo in {base_dir}!")
    print(f"Train: {n_train_n} NORMAL + {n_train_p} PNEUMONIA = {n_train_n + n_train_p}")
    print(f"Val:   {n_val_n} NORMAL + {n_val_p} PNEUMONIA = {n_val_n + n_val_p} (incluso {n_orig_val} da original val)")
    print(f"Sbilanciamento Train: {n_train_p / n_train_n:.2f}:1 (PNEUMONIA:NORMAL)")

    return train_dir, val_dir, test_dir, n_train_n, n_train_p

def get_dataloaders(train_dir, val_dir, test_dir, img_size=128, batch_size=16):
    """
    Restituisce i DataLoader per ResNet (RGB convertiti nativamente da ImageFolder 
    o grayscale se necessari modificando la transform).
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

def get_gan_dataloader(train_dir, img_size=128, batch_size=64):
    """
    Restituisce il DataLoader per il GAN WGAN-GP (1 canale Grayscale, 128x128).
    """
    gan_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    gan_dataset = datasets.ImageFolder(root=train_dir, transform=gan_transform)
    gan_loader = DataLoader(gan_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return gan_loader, gan_dataset.classes
