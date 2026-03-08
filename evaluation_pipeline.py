
# ============================================================
# PIPELINE COMPLETA: Phase 1 → GAN Training → Phase 3
# ============================================================
# Questo file contiene l'intero workflow:
#  Cella 1:  Setup, imports, estrazione dataset
#  Cella 2:  Split Train/Val/Test (80/20 dal train originale)
#  Cella 3:  DataLoaders e trasformazioni
#  Cella 4:  Phase 1 — ResNet18 Baseline (dataset originale sbilanciato)
#  Cella 5:  Valutazione Phase 1 sul Test Set
#  Cella 6:  WGAN-GP — Definizione architettura
#  Cella 7:  WGAN-GP — Training
#  Cella 8:  Generazione immagini sintetiche per bilanciamento
#  Cella 9:  Creazione dataset augmented (reale + sintetico)
#  Cella 10: Phase 3 — ResNet18 su dataset augmented
#  Cella 11: Valutazione Phase 3 sul Test Set
#  Cella 12: Confronto Phase 1 vs Phase 3


# ============================================================
# CELLA 1 — Setup
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad
import numpy as np
import random
import os
import shutil
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from google.colab import drive
import zipfile

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

# Monta Drive e scompatta
drive.mount('/content/drive')
zip_path = '/content/drive/MyDrive/ProgettoMLVM/chest_xray.zip'
extract_path = './original_dataset'

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Dataset estratto!")


# ============================================================
# CELLA 2 — Split Dataset (Train 80% / Val 20% / Test intatto)
# ============================================================

orig_train_norm = 'original_dataset/chest_xray/train/NORMAL'
orig_train_pneu = 'original_dataset/chest_xray/train/PNEUMONIA'
orig_test = 'original_dataset/chest_xray/test'

base_dir = 'project_dataset'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

for cat in ['NORMAL', 'PNEUMONIA']:
    os.makedirs(os.path.join(train_dir, cat), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cat), exist_ok=True)

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

n_train_n, n_val_n = split_and_copy(orig_train_norm,
    os.path.join(train_dir, 'NORMAL'), os.path.join(val_dir, 'NORMAL'))
n_train_p, n_val_p = split_and_copy(orig_train_pneu,
    os.path.join(train_dir, 'PNEUMONIA'), os.path.join(val_dir, 'PNEUMONIA'))

# Aggiungi le 16 immagini della validation originale (troppo poche da sole, ma utili)
orig_val = 'original_dataset/chest_xray/val'
n_orig_val = 0
for cat in ['NORMAL', 'PNEUMONIA']:
    orig_val_cat = os.path.join(orig_val, cat)
    if os.path.exists(orig_val_cat):
        for f in os.listdir(orig_val_cat):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                shutil.copy(os.path.join(orig_val_cat, f),
                           os.path.join(val_dir, cat, f))
                n_orig_val += 1
                if cat == 'NORMAL':
                    n_val_n += 1
                else:
                    n_val_p += 1

print(f"Train:  {n_train_n} NORMAL + {n_train_p} PNEUMONIA = {n_train_n + n_train_p}")
print(f"Val:    {n_val_n} NORMAL + {n_val_p} PNEUMONIA = {n_val_n + n_val_p} (di cui {n_orig_val} dalla val originale)")
print(f"Sbilanciamento: {n_train_p / n_train_n:.2f}:1 (PNEUMONIA:NORMAL)")


# ============================================================
# CELLA 3 — DataLoaders
# ============================================================

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"Classi: {train_dataset.classes}")
print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")


# ============================================================
# CELLA 4 — Phase 1: ResNet18 Baseline
# ============================================================

def train_resnet(train_loader, val_loader, epochs=5, lr=0.001, tag="Phase1"):
    """Allena ResNet18 e restituisce history + percorso checkpoint."""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    ckpt_path = f'best_model_{tag}.pth'
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*50}")
    print(f"Training {tag}")
    print(f"{'='*50}")

    for epoch in range(epochs):
        # Train
        model.train()
        rl = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            rl += loss.item()
        avg_tl = rl / len(train_loader)

        # Validation
        model.eval()
        vl, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vl += criterion(out, y).item()
                _, pred = torch.max(out, 1)
                total += y.size(0)
                correct += (pred == y).sum().item()
        avg_vl = vl / len(val_loader)
        acc = 100 * correct / total

        history['train_loss'].append(avg_tl)
        history['val_loss'].append(avg_vl)
        history['val_acc'].append(acc)

        saved = ""
        if avg_vl < best_val_loss:
            best_val_loss = avg_vl
            torch.save(model.state_dict(), ckpt_path)
            saved = " 🌟"

        print(f"  Epoch {epoch+1}/{epochs} | TL: {avg_tl:.4f} | VL: {avg_vl:.4f} | VA: {acc:.2f}%{saved}")

    return model, history, ckpt_path

# --- Lancia Phase 1 ---
model_p1, hist_p1, ckpt_p1 = train_resnet(train_loader, val_loader, epochs=5, tag="Phase1")


# ============================================================
# CELLA 5 — Valutazione Phase 1 sul Test Set
# ============================================================

def evaluate_on_test(model, ckpt_path, test_loader, class_names, tag="Phase1"):
    """Valuta il modello sul test set e restituisce report + confusion matrix."""
    model.load_state_dict(torch.load(ckpt_path))
    model = model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x), 1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    print(f"\n{'='*50}")
    print(f"RISULTATI {tag}")
    print(f"{'='*50}")
    report = classification_report(all_labels, all_preds,
                                   target_names=class_names, output_dict=True)
    print(classification_report(all_labels, all_preds, target_names=class_names))

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'Confusion Matrix — {tag}')
    plt.tight_layout()
    plt.savefig(f'cm_{tag}.png')
    plt.show()

    return report, cm

report_p1, cm_p1 = evaluate_on_test(model_p1, ckpt_p1, test_loader,
                                     train_dataset.classes, "Phase1")


# ============================================================
# CELLA 6 — WGAN-GP: Architettura (Generator + Critic)
# ============================================================

# Parametri GAN
gan_img_size = 128    # Risoluzione aumentata (era 64)
gan_nz = 100
gan_n_class = 2
gan_nc = 1
gan_lambda_gp = 10
gan_n_critic = 5
gan_lr = 0.0001
gan_epochs = 100      # Più epoche per convergere a risoluzione maggiore


class Generator(nn.Module):
    """
    Generator 128×128: aggiunto un layer rispetto alla versione 64×64.
    Flow: z(1×1) → 4×4 → 8×8 → 16×16 → 32×32 → 64×64 → 128×128
    """
    def __init__(self, d=128):
        super().__init__()
        # Input branches (entrambe producono 4×4)
        self.deconv1_1 = nn.ConvTranspose2d(gan_nz, d*4, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*4)
        self.deconv1_2 = nn.ConvTranspose2d(gan_n_class, d*4, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*4)
        # d*8 → d*4 (4×4 → 8×8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        # d*4 → d*2 (8×8 → 16×16)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        # d*2 → d (16×16 → 32×32)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        # d → d//2 (32×32 → 64×64)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        # d//2 → nc (64×64 → 128×128) ← NUOVO
        self.deconv6 = nn.ConvTranspose2d(d//2, gan_nc, 4, 2, 1)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], (nn.ConvTranspose2d, nn.Conv2d)):
                self._modules[m].weight.data.normal_(mean, std)

    def forward(self, z, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(z)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        return torch.tanh(self.deconv6(x))


class Critic(nn.Module):
    """
    Critic 128×128: aggiunto un layer rispetto alla versione 64×64.
    Flow: 128×128 → 64×64 → 32×32 → 16×16 → 8×8 → 4×4 → 1×1
    """
    def __init__(self, d=128):
        super().__init__()
        # Input branches (128×128 → 64×64)
        self.conv1_1 = nn.Conv2d(gan_nc, d//2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(gan_n_class, d//2, 4, 2, 1)
        # d → d*2 (64×64 → 32×32)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_ln = nn.LayerNorm([d*2, 32, 32])
        # d*2 → d*4 (32×32 → 16×16)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_ln = nn.LayerNorm([d*4, 16, 16])
        # d*4 → d*8 (16×16 → 8×8)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_ln = nn.LayerNorm([d*8, 8, 8])
        # d*8 → d*8 (8×8 → 4×4) ← NUOVO
        self.conv5 = nn.Conv2d(d*8, d*8, 4, 2, 1)
        self.conv5_ln = nn.LayerNorm([d*8, 4, 4])
        # d*8 → 1 (4×4 → 1×1)
        self.conv6 = nn.Conv2d(d*8, 1, 4, 1, 0)

    def weight_init(self, mean, std):
        for m in self._modules:
            if isinstance(self._modules[m], (nn.ConvTranspose2d, nn.Conv2d)):
                self._modules[m].weight.data.normal_(mean, std)

    def forward(self, img, label):
        x = F.leaky_relu(self.conv1_1(img), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_ln(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_ln(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_ln(self.conv4(x)), 0.2)
        x = F.leaky_relu(self.conv5_ln(self.conv5(x)), 0.2)
        return self.conv6(x)


def compute_gp(D, real, fake, real_lbl, fake_lbl):
    bs = real.size(0)
    alpha = torch.rand(bs, 1, 1, 1).to(device).expand_as(real)
    interp = (alpha * real.data + (1-alpha) * fake.data).requires_grad_(True)
    alpha_l = torch.rand(bs, 1, 1, 1).to(device).expand_as(real_lbl)
    interp_l = alpha_l * real_lbl.data + (1-alpha_l) * fake_lbl.data
    d_interp = D(interp, interp_l)
    grads = torch_grad(outputs=d_interp, inputs=interp,
                       grad_outputs=torch.ones_like(d_interp),
                       create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    gn = torch.sqrt(torch.sum(grads**2, dim=1) + 1e-12)
    return gan_lambda_gp * ((gn - 1)**2).mean(), gn.mean().item()


print("Architettura GAN 128×128 definita ✅")


# ============================================================
# CELLA 7 — WGAN-GP: Training
# ============================================================

# Dataset GAN (grayscale, 64x64, normalizzato [-1,1])
gan_transform = transforms.Compose([
    transforms.Resize((gan_img_size, gan_img_size)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,))
])

gan_dataset = datasets.ImageFolder(root=train_dir, transform=gan_transform)
gan_loader = DataLoader(gan_dataset, batch_size=64, shuffle=True, drop_last=True)

print(f"GAN dataset: {len(gan_dataset)} immagini, {len(gan_loader)} batch/epoca")

# Init modelli
G = Generator(128).to(device)
D = Critic(128).to(device)
G.weight_init(0.0, 0.02)
D.weight_init(0.0, 0.02)

G_opt = optim.Adam(G.parameters(), lr=gan_lr, betas=(0.0, 0.9))
D_opt = optim.Adam(D.parameters(), lr=gan_lr, betas=(0.0, 0.9))

# Label processing
onehot = torch.eye(gan_n_class).view(gan_n_class, gan_n_class, 1, 1).to(device)
fill = torch.zeros([gan_n_class, gan_n_class, gan_img_size, gan_img_size]).to(device)
for i in range(gan_n_class):
    fill[i, i, :, :] = 1

print(f"\nTraining WGAN-GP ({gan_epochs} epoche, n_critic={gan_n_critic})...")

# --- Fixed noise per visualizzare l'evoluzione ---
num_samples = 5
fixed_z = torch.randn(num_samples * 2, gan_nz, 1, 1).to(device)  # Stesso noise per tutte le epoche
fixed_labels = torch.cat([
    onehot[torch.zeros(num_samples, dtype=torch.long).to(device)],   # NORMAL
    onehot[torch.ones(num_samples, dtype=torch.long).to(device)]     # PNEUMONIA
])

def show_gan_samples(epoch):
    """Mostra una griglia 2×5: 5 NORMAL + 5 PNEUMONIA."""
    G.eval()
    with torch.no_grad():
        imgs = G(fixed_z, fixed_labels)
    G.train()

    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5))
    for cls_idx, cls_name in enumerate(['NORMAL', 'PNEUMONIA']):
        for j in range(num_samples):
            idx = cls_idx * num_samples + j
            axes[cls_idx, j].imshow(imgs[idx, 0].cpu().numpy(), cmap='gray')
            axes[cls_idx, j].axis('off')
            if j == 0:
                axes[cls_idx, j].set_ylabel(cls_name, fontsize=10)
    plt.suptitle(f'GAN Samples — Epoch {epoch}', fontsize=13)
    plt.tight_layout()
    plt.show()

gan_start = time.time()

for epoch in range(gan_epochs):
    d_losses, g_losses = [], []

    # LR decay (adattato per 100 epoche)
    if (epoch+1) == 60:
        for pg in G_opt.param_groups + D_opt.param_groups: pg['lr'] /= 5
    if (epoch+1) == 80:
        for pg in G_opt.param_groups + D_opt.param_groups: pg['lr'] /= 5

    for batch_idx, (x_, y_) in enumerate(gan_loader):
        mb = x_.size(0)
        x_, y_ = x_.to(device), y_.to(device)
        y_fill = fill[y_]

        # --- Train Critic ---
        D.zero_grad()
        D_real = D(x_, y_fill).squeeze().mean()

        z = torch.randn(mb, gan_nz, 1, 1).to(device)
        y_gen = torch.randint(0, gan_n_class, (mb,)).to(device)
        fake = G(z, onehot[y_gen])
        D_fake = D(fake.detach(), fill[y_gen]).squeeze().mean()

        gp, _ = compute_gp(D, x_, fake.detach(), y_fill, fill[y_gen])
        d_loss = D_fake - D_real + gp
        d_loss.backward()
        D_opt.step()
        d_losses.append(d_loss.item())

        # --- Train Generator ---
        if (batch_idx+1) % gan_n_critic == 0:
            G.zero_grad()
            z = torch.randn(mb, gan_nz, 1, 1).to(device)
            y_gen = torch.randint(0, gan_n_class, (mb,)).to(device)
            g_loss = -D(G(z, onehot[y_gen]), fill[y_gen]).squeeze().mean()
            g_loss.backward()
            G_opt.step()
            g_losses.append(g_loss.item())

    # --- Log ogni 5 epoche ---
    if (epoch+1) % 5 == 0 or epoch == 0:
        print(f"  [{epoch+1}/{gan_epochs}] W_dist: {-np.mean(d_losses):.1f}, "
              f"G_loss: {np.mean(g_losses):.1f}")
        show_gan_samples(epoch + 1)

gan_time = (time.time() - gan_start) / 60
print(f"\nGAN training completato in {gan_time:.1f} minuti!")
torch.save(G.state_dict(), '/content/drive/MyDrive/ProgettoMLVM/G_pipeline.pth')


# ============================================================
# CELLA 8 — Generazione Immagini per Bilanciamento
# ============================================================
# Strategia: portiamo NORMAL allo stesso numero di PNEUMONIA
# Es: Train ha 1073 NORMAL e 3100 PNEUMONIA → generiamo ~2027 NORMAL sintetici

num_gen_normal = n_train_p - n_train_n     # Colma il gap
num_gen_pneumonia = 0                       # Pneumonia è già la classe maggioritaria

print(f"\n{'='*50}")
print(f"BILANCIAMENTO DATASET")
print(f"{'='*50}")
print(f"  Train originale: {n_train_n} NORMAL, {n_train_p} PNEUMONIA")
print(f"  Gap da colmare: {num_gen_normal} immagini NORMAL sintetiche")
print(f"  Dopo augmentation: {n_train_n + num_gen_normal} NORMAL, {n_train_p} PNEUMONIA (1:1)")

syn_dir = 'synthetic_images'
os.makedirs(os.path.join(syn_dir, 'NORMAL'), exist_ok=True)
os.makedirs(os.path.join(syn_dir, 'PNEUMONIA'), exist_ok=True)

G.eval()
with torch.no_grad():
    for cls_idx, cls_name, num_gen in [(0, 'NORMAL', num_gen_normal),
                                        (1, 'PNEUMONIA', num_gen_pneumonia)]:
        if num_gen <= 0:
            print(f"  {cls_name}: nessuna generazione necessaria")
            continue

        generated = 0
        while generated < num_gen:
            batch_sz = min(64, num_gen - generated)
            z = torch.randn(batch_sz, gan_nz, 1, 1).to(device)
            labels = onehot[torch.full((batch_sz,), cls_idx, dtype=torch.long).to(device)]
            fakes = G(z, labels)

            for i in range(batch_sz):
                img = fakes[i].cpu()
                img = (img + 1) / 2.0  # [-1,1] → [0,1]
                img_pil = transforms.ToPILImage()(img)
                # Converti in RGB (già 128×128, stessa risoluzione del classificatore)
                img_rgb = img_pil.convert('RGB')
                img_rgb.save(os.path.join(syn_dir, cls_name, f'syn_{cls_name}_{generated+i}.png'))

            generated += batch_sz

        print(f"  {cls_name}: generate {num_gen} immagini sintetiche")

print("Generazione completata! ✅")


# ============================================================
# CELLA 9 — Creazione Dataset Augmented (Reale + Sintetico)
# ============================================================

aug_dir = 'augmented_dataset'
aug_train_dir = os.path.join(aug_dir, 'train')

if os.path.exists(aug_dir):
    shutil.rmtree(aug_dir)

# Copia train originale
shutil.copytree(train_dir, aug_train_dir)

# Aggiungi immagini sintetiche
for cls_name in ['NORMAL', 'PNEUMONIA']:
    syn_cls_dir = os.path.join(syn_dir, cls_name)
    if os.path.exists(syn_cls_dir):
        for f in os.listdir(syn_cls_dir):
            shutil.copy(os.path.join(syn_cls_dir, f),
                       os.path.join(aug_train_dir, cls_name, f))

# Conta
aug_n = len(os.listdir(os.path.join(aug_train_dir, 'NORMAL')))
aug_p = len(os.listdir(os.path.join(aug_train_dir, 'PNEUMONIA')))
print(f"\nDataset Augmented creato:")
print(f"  NORMAL:    {n_train_n} reali + {num_gen_normal} sintetici = {aug_n}")
print(f"  PNEUMONIA: {n_train_p} reali + {num_gen_pneumonia} sintetici = {aug_p}")
print(f"  Rapporto: {aug_p / aug_n:.2f}:1")


# ============================================================
# CELLA 10 — Phase 3: ResNet18 su Dataset Augmented
# ============================================================

aug_train_dataset = datasets.ImageFolder(root=aug_train_dir, transform=transform)
aug_train_loader = DataLoader(aug_train_dataset, batch_size=16, shuffle=True)

print(f"Augmented train: {len(aug_train_dataset)} immagini")

# Riusa la stessa validation (NON augmentata — deve essere reale!)
model_p3, hist_p3, ckpt_p3 = train_resnet(aug_train_loader, val_loader, epochs=5, tag="Phase3")


# ============================================================
# CELLA 11 — Valutazione Phase 3 sul Test Set
# ============================================================

report_p3, cm_p3 = evaluate_on_test(model_p3, ckpt_p3, test_loader,
                                     train_dataset.classes, "Phase3")


# ============================================================
# CELLA 12 — Confronto Phase 1 vs Phase 3
# ============================================================

print("\n" + "=" * 60)
print("CONFRONTO FINAL: Phase 1 (Baseline) vs Phase 3 (Augmented)")
print("=" * 60)

metrics = ['precision', 'recall', 'f1-score']
for cls in train_dataset.classes:
    print(f"\n  {cls}:")
    for m in metrics:
        v1 = report_p1[cls][m]
        v3 = report_p3[cls][m]
        diff = v3 - v1
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"    {m:12s}: {v1:.4f} → {v3:.4f}  ({arrow} {abs(diff):.4f})")

# Accuracy complessiva
acc_p1 = report_p1['accuracy']
acc_p3 = report_p3['accuracy']
diff_acc = acc_p3 - acc_p1
arrow = "↑" if diff_acc > 0 else "↓"
print(f"\n  Overall Accuracy: {acc_p1:.4f} → {acc_p3:.4f}  ({arrow} {abs(diff_acc):.4f})")

# F1 macro
f1_p1 = report_p1['macro avg']['f1-score']
f1_p3 = report_p3['macro avg']['f1-score']
diff_f1 = f1_p3 - f1_p1
arrow = "↑" if diff_f1 > 0 else "↓"
print(f"  Macro F1:        {f1_p1:.4f} → {f1_p3:.4f}  ({arrow} {abs(diff_f1):.4f})")

# Plot confronto curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(hist_p1['train_loss'], label='Phase 1', marker='o')
axes[0].plot(hist_p3['train_loss'], label='Phase 3', marker='s')
axes[0].set_title('Train Loss')
axes[0].legend(); axes[0].grid(True)

axes[1].plot(hist_p1['val_loss'], label='Phase 1', marker='o')
axes[1].plot(hist_p3['val_loss'], label='Phase 3', marker='s')
axes[1].set_title('Val Loss')
axes[1].legend(); axes[1].grid(True)

axes[2].plot(hist_p1['val_acc'], label='Phase 1', marker='o')
axes[2].plot(hist_p3['val_acc'], label='Phase 3', marker='s')
axes[2].set_title('Val Accuracy')
axes[2].legend(); axes[2].grid(True)

plt.suptitle('Phase 1 vs Phase 3 — Training Curves', fontsize=14)
plt.tight_layout()
plt.savefig('comparison_p1_vs_p3.png')
plt.show()

# Confronto confusion matrix side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_p1, annot=True, fmt='d', cmap='Blues',
            xticklabels=train_dataset.classes, yticklabels=train_dataset.classes, ax=ax1)
ax1.set_title('Phase 1 (Baseline)')
ax1.set_xlabel('Predicted'); ax1.set_ylabel('True')

sns.heatmap(cm_p3, annot=True, fmt='d', cmap='Greens',
            xticklabels=train_dataset.classes, yticklabels=train_dataset.classes, ax=ax2)
ax2.set_title('Phase 3 (Augmented)')
ax2.set_xlabel('Predicted'); ax2.set_ylabel('True')

plt.suptitle('Confusion Matrix Comparison', fontsize=14)
plt.tight_layout()
plt.savefig('cm_comparison.png')
plt.show()

print("\n✅ Pipeline completata!")
