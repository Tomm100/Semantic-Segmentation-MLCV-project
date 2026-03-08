import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

from models.resnet import get_resnet_classifier

def train_resnet(train_loader, val_loader, device, epochs=5, lr=0.001, tag="Phase1"):
    """
    Allena ResNet18 (Feature Extractor / Classifier).
    Salva il checkpoint migliore in base alla Validation Loss.
    """
    model = get_resnet_classifier(num_classes=2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    ckpt_path = f'best_model_{tag}.pth'
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    print(f"\n{'='*50}\nTraining {tag}\n{'='*50}")

    for epoch in range(epochs):
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


def train_wgangp(G, D, gan_loader, device, compute_gp_fn, epochs=100, lr=0.0001, n_critic=5, nz=100, n_class=2, out_dir='gan_samples', ckpt_path='/content/drive/MyDrive/ProgettoMLVM/G_pipeline.pth'):
    """
    Allena WGAN-GP per la generazione di immagini.
    Salva una griglia di sample ogni 5 epoche in `out_dir` per valutarne l'evoluzione.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    # Inizializzazione pesi
    G.weight_init(0.0, 0.02)
    D.weight_init(0.0, 0.02)

    G_opt = optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    D_opt = optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    # Tensors per label condizionali spaziali
    img_size = gan_loader.dataset[0][0].shape[1] # Assumendo (C, H, W) e img quadrate
    onehot = torch.eye(n_class).view(n_class, n_class, 1, 1).to(device)
    fill = torch.zeros([n_class, n_class, img_size, img_size]).to(device)
    for i in range(n_class):
        fill[i, i, :, :] = 1

    print(f"\nTraining WGAN-GP ({epochs} epoche, n_critic={n_critic})...")
    
    # Setup log sample immagini (fixed noise per vedere l'evoluzione)
    num_samples = 5
    fixed_z = torch.randn(num_samples * 2, nz, 1, 1).to(device)
    fixed_labels = torch.cat([
        onehot[torch.zeros(num_samples, dtype=torch.long).to(device)],   # NORMAL
        onehot[torch.ones(num_samples, dtype=torch.long).to(device)]     # PNEUMONIA
    ])

    def save_gan_samples(epoch):
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
                if j == 0: axes[cls_idx, j].set_ylabel(cls_name, fontsize=10)
        plt.suptitle(f'GAN Samples — Epoch {epoch}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'samples_epoch_{epoch:03d}.png'))
        plt.close(fig)

    gan_start = time.time()

    for epoch in range(epochs):
        d_losses, g_losses = [], []

        # LR decay schedule
        if (epoch+1) == 60:
            for pg in G_opt.param_groups + D_opt.param_groups: pg['lr'] /= 5
        if (epoch+1) == 80:
            for pg in G_opt.param_groups + D_opt.param_groups: pg['lr'] /= 5

        for batch_idx, (x_, y_) in enumerate(gan_loader):
            mb = x_.size(0)
            x_, y_ = x_.to(device), y_.to(device)
            y_fill = fill[y_]

            # Train Critic
            D.zero_grad()
            D_real = D(x_, y_fill).squeeze().mean()
            
            z = torch.randn(mb, nz, 1, 1).to(device)
            y_gen = torch.randint(0, n_class, (mb,)).to(device)
            fake = G(z, onehot[y_gen])
            D_fake = D(fake.detach(), fill[y_gen]).squeeze().mean()
            
            gp, _ = compute_gp_fn(D, x_, fake.detach(), y_fill, fill[y_gen], device)
            d_loss = D_fake - D_real + gp
            d_loss.backward()
            D_opt.step()
            d_losses.append(d_loss.item())

            # Train Generator
            if (batch_idx+1) % n_critic == 0:
                G.zero_grad()
                z = torch.randn(mb, nz, 1, 1).to(device)
                y_gen = torch.randint(0, n_class, (mb,)).to(device)
                g_loss = -D(G(z, onehot[y_gen]), fill[y_gen]).squeeze().mean()
                g_loss.backward()
                G_opt.step()
                g_losses.append(g_loss.item())

        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"  [{epoch+1}/{epochs}] W_dist: {-np.mean(d_losses):.1f}, G_loss: {np.mean(g_losses):.1f}")
            save_gan_samples(epoch + 1)

    gan_time = (time.time() - gan_start) / 60
    print(f"\nGAN training completato in {gan_time:.1f} minuti!")
    
    # Save model
    torch.save(G.state_dict(), ckpt_path)
    return G, ckpt_path
