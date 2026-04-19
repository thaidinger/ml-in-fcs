import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw
from tqdm import tqdm
import math

# =============================================================================
# CONFIGURATION
# =============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
L_MIN, L_MAX, K, DIFFUSION_STEPS = 10, 21, 5, 100
BATCH_SIZE, GEN_LR, EVOL_LR = 32, 5e-4, 4e-4
EPOCHS_GEN, EPOCHS_EVOL = 30, 50

# =============================================================================
# 1. PATTERN RECOGNITION MODULE (Sec 4.1, Algorithm 1)
# =============================================================================
class SISC:
    def __init__(self, K=K, l_min=L_MIN, l_max=L_MAX, max_iters=15):
        self.K = K
        self.l_min = l_min
        self.l_max = l_max
        self.max_iters = max_iters
        self.centroids = []

    def _dtw_dist(self, x, y):
        # x, y: 1D numpy arrays
        dist, _ = fastdtw(x.reshape(-1, 1), y.reshape(-1, 1))
        return dist

    def _wise_init(self, X):
        candidates = [X[t:t+self.l_max] for t in range(len(X)-self.l_max+1)]
        idx0 = np.random.randint(0, len(candidates))
        self.centroids = [candidates[idx0]]
        
        for _ in range(self.K - 1):
            dists = np.array([min(self._dtw_dist(c, cent) for cent in self.centroids) for c in candidates])
            probs = dists / dists.sum()
            idx = np.random.choice(len(candidates), p=probs)
            self.centroids.append(candidates[idx])

    def fit(self, X):
        self._wise_init(X)
        
        for _ in range(self.max_iters):
            # Greedy Segmentation
            segments = []
            t = 0
            while t <= len(X) - self.l_min:
                best_l, best_dist, best_cidx = None, float('inf'), None
                end = min(len(X), t + self.l_max + 1)
                for l in range(self.l_min, end - t + 1):
                    seg = X[t:t+l]
                    for c_idx, cent in enumerate(self.centroids):
                        d = self._dtw_dist(seg, cent)
                        if d < best_dist:
                            best_dist, best_l, best_cidx = d, l, c_idx
                segments.append((X[t:t+best_l], best_cidx))
                t += best_l

            # Update Centroids
            new_centroids = []
            for c in range(self.K):
                clust_segs = [seg for seg, idx in segments if idx == c]
                if not clust_segs:
                    new_centroids.append(self.centroids[c])
                    continue
                # Interpolate to l_max for stable averaging
                aligned = []
                for seg in clust_segs:
                    tensor_seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    aligned.append(F.interpolate(tensor_seg, size=self.l_max, mode='linear').squeeze().numpy())
                new_centroids.append(np.mean(aligned, axis=0))
            self.centroids = new_centroids
            
        self.segment_data = [seg for seg, _ in segments]
        self.segment_labels = [idx for _, idx in segments]
        return self.segment_data, self.segment_labels

# =============================================================================
# 2. PATTERN GENERATION MODULE (Sec 4.2)
# =============================================================================
class TCNBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        res = x
        out = F.gelu(self.conv1(x))
        out = self.dropout(out)
        out = self.conv2(out)
        return F.gelu(out + res)

class ScalingAutoencoder(nn.Module):
    """Maps variable-length segments to/from fixed-length (Sec 4.2)"""
    def __init__(self, fixed_len=L_MAX, hidden_dim=32):
        super().__init__()
        self.fixed_len = fixed_len
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool1d(fixed_len),
            nn.Conv1d(1, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_dim, 1, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def encode(self, x):
        # x: (B, 1, L_var) -> (B, hidden, fixed_len)
        return self.encoder(x)

    def decode(self, h, target_len):
        # h: (B, hidden, fixed_len) -> (B, 1, target_len)
        out = self.decoder(h)
        return F.interpolate(out, size=target_len, mode='linear')

class PatternConditionedDiffusion(nn.Module):
    """DDPM Denoiser conditioned on pattern centroids (Eq 3, 4)"""
    def __init__(self, seq_len=L_MAX, channels=64, n_steps=DIFFUSION_STEPS):
        super().__init__()
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.channels = channels
        self.beta = torch.linspace(1e-4, 0.02, n_steps)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
        
        self.time_emb = nn.Embedding(n_steps, channels)
        self.pattern_emb = nn.Linear(seq_len, channels)
        # Fixed: input is now (B, 1 + 2*channels, seq_len) after proper concatenation
        self.input_proj = nn.Conv1d(1 + channels*2, channels, kernel_size=1)
        self.blocks = nn.Sequential(*[TCNBlock(channels) for _ in range(6)])
        self.out_conv = nn.Conv1d(channels, 1, kernel_size=3, padding=1)

    def _get_beta_schedule(self, t): return self.beta[t]

    def forward(self, x, t, pattern_cond):
        # x: (B, 1, L), t: (B,), pattern_cond: (B, L)
        B = x.shape[0]
        
        # Fix: Properly expand embeddings to match sequence length
        t_emb = self.time_emb(t)  # (B, channels)
        t_emb = t_emb.unsqueeze(2).expand(-1, -1, self.seq_len)  # (B, channels, L)
        
        p_emb = self.pattern_emb(pattern_cond)  # (B, channels)
        p_emb = p_emb.unsqueeze(2).expand(-1, -1, self.seq_len)  # (B, channels, L)
        
        # Now concatenate along channel dimension
        h = torch.cat([x, t_emb, p_emb], dim=1)  # (B, 1+2*channels, L)
        h = self.input_proj(h)  # (B, channels, L)
        h = self.blocks(h)
        return self.out_conv(h)

    def q_sample(self, x_start, t, noise=None):
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t])[:, None, None]
        return sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise, noise

    def p_sample(self, x, t, pattern_cond):
        if (t == 0).all(): return x
        beta_t = self.beta[t][:, None, None]
        alpha_t = self.alpha[t][:, None, None]
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t])[:, None, None]
        
        epsilon_theta = self.forward(x, t, pattern_cond)
        mu = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / sqrt_alpha_bar) * epsilon_theta)
        sigma = torch.sqrt(beta_t) if t[0] > 0 else 0
        return mu + sigma * torch.randn_like(x) if t[0] > 0 else mu

class PatternGenerationModule(nn.Module):
    def __init__(self, seq_len=L_MAX):
        super().__init__()
        self.ae = ScalingAutoencoder(fixed_len=seq_len)
        self.diffusion = PatternConditionedDiffusion(seq_len=seq_len)
        
    def loss(self, x_fixed, t, pattern_idx, centroids):
        """
        x_fixed: (B, 1, L_MAX) - pre-interpolated to fixed length
        t: (B,) diffusion time steps
        pattern_idx: (B,) cluster labels
        centroids: list of numpy arrays, each length L_MAX
        """
        # Add noise
        noise = torch.randn_like(x_fixed)
        x_noisy, added_noise = self.diffusion.q_sample(x_fixed, t, noise)
        
        # Fixed: Convert centroids to numpy array first, then to tensor
        pattern_cond_np = np.array([centroids[i.item()] for i in pattern_idx.cpu()])
        pattern_cond = torch.from_numpy(pattern_cond_np).float().to(x_fixed.device)
        
        # Predict noise
        epsilon_pred = self.diffusion(x_noisy, t, pattern_cond)
        
        # Eq 5: Denoising loss
        denoise_loss = F.mse_loss(epsilon_pred, added_noise)
        return denoise_loss

# =============================================================================
# 3. PATTERN EVOLUTION MODULE (Sec 4.3)
# =============================================================================
class PatternEvolutionNetwork(nn.Module):
    def __init__(self, K=K, hidden_dim=64):
        super().__init__()
        self.K = K
        self.fc = nn.Sequential(
            nn.Linear(K + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.out_pattern = nn.Linear(hidden_dim, K)
        self.out_alpha = nn.Linear(hidden_dim, 1)
        self.out_beta = nn.Linear(hidden_dim, 1)

    def forward(self, p_current, alpha_current, beta_current):
        p_onehot = F.one_hot(p_current.long(), self.K).float()
        state = torch.cat([p_onehot, alpha_current.unsqueeze(1), beta_current.unsqueeze(1)], dim=1)
        h = self.fc(state)
        return self.out_pattern(h), self.out_alpha(h).squeeze(-1), self.out_beta(h).squeeze(-1)

# =============================================================================
# 4. ORCHESTRATION & SAMPLING (Sec 4.4, Algorithm 2)
# =============================================================================
class FTSDiffusion:
    def __init__(self, device=DEVICE):
        self.device = device
        self.sisc = SISC()
        self.gen_module = PatternGenerationModule().to(device)
        self.evol_module = PatternEvolutionNetwork().to(device)
        self.centroids = None
        
    def train_sisc(self, X):
        print("▶ Training SISC Pattern Recognition...")
        self.segment_data, self.segment_labels = self.sisc.fit(X)
        self.centroids = self.sisc.centroids
        print(f"  Found {len(self.centroids)} scale-invariant patterns.")
        
    def _get_scaling_factors(self):
        alphas = [len(seg)/L_MAX for seg in self.segment_data]
        betas = [np.std(seg) + 1e-6 for seg in self.segment_data]
        return np.array(alphas), np.array(betas)

    def train_generation(self, epochs=EPOCHS_GEN):
        print("▶ Training Pattern Generation Module...")
        optimizer = torch.optim.Adam(self.gen_module.parameters(), lr=GEN_LR)
        
        for epoch in range(epochs):
            self.gen_module.train()
            total_loss = 0
            steps = max(1, len(self.segment_data) // BATCH_SIZE)
            for _ in range(steps):
                idx = list(np.random.choice(len(self.segment_data), BATCH_SIZE))
                batch_segments = [self.segment_data[i] for i in idx]
                batch_labels = [self.segment_labels[i] for i in idx]
                
                # Interpolate variable-length segments to fixed L_MAX before batching
                batch_x_list = []
                for seg in batch_segments:
                    t_seg = torch.tensor(seg, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    t_fixed = F.interpolate(t_seg, size=L_MAX, mode='linear', align_corners=False)
                    batch_x_list.append(t_fixed)
                
                batch_x = torch.cat(batch_x_list, dim=0).to(self.device)
                batch_labels = torch.tensor(batch_labels, dtype=torch.long).to(self.device)
                t = torch.randint(0, DIFFUSION_STEPS, (BATCH_SIZE,), device=self.device)
                
                optimizer.zero_grad()
                loss = self.gen_module.loss(batch_x, t, batch_labels, self.centroids)
                loss.backward() 
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0: 
                print(f"  Gen Loss Epoch {epoch}: {total_loss/steps:.4f}")

    def train_evolution(self, epochs=EPOCHS_EVOL):
        print("▶ Training Pattern Evolution Network...")
        optimizer = torch.optim.Adam(self.evol_module.parameters(), lr=EVOL_LR)
        alphas, betas = self._get_scaling_factors()
        
        # Build transition dataset
        trans_data = []
        for i in range(len(self.segment_labels)-1):
            trans_data.append((self.segment_labels[i], alphas[i], betas[i], 
                               self.segment_labels[i+1], alphas[i+1], betas[i+1]))
        trans_data = np.array(trans_data)
        
        for epoch in range(epochs):
            idx = np.random.choice(len(trans_data), BATCH_SIZE)
            batch = trans_data[idx]
            p_curr, a_curr, b_curr, p_next, a_next, b_next = [
                torch.tensor(batch[:, i], dtype=torch.float32).to(self.device) for i in range(6)
            ]
            
            optimizer.zero_grad()
            logits, pred_a, pred_b = self.evol_module(p_curr, a_curr, b_curr)
            loss = F.cross_entropy(logits, p_next.long()) + F.mse_loss(pred_a, a_next) + F.mse_loss(pred_b, b_next)
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0: print(f"  Evol Loss Epoch {epoch}: {loss.item():.4f}")

    def sample(self, init_seg, target_len=1000):
        print("▶ Synthesizing Financial Time Series...")
        synth = list(init_seg)
        curr_p = torch.tensor([0], dtype=torch.float32, device=self.device)
        curr_a, curr_b = torch.tensor([1.0], device=self.device), torch.tensor([1.0], device=self.device)
        self.gen_module.eval()
        self.evol_module.eval()
        
        steps = 0
        while len(synth) < target_len and steps < 200:
            with torch.no_grad():
                logits, pred_a, pred_b = self.evol_module(curr_p, curr_a, curr_b)
                next_p = torch.argmax(logits).item()
                next_a, next_b = pred_a.item(), pred_b.item()
                
                # Denoise
                noise = torch.randn(1, 1, L_MAX, device=self.device)
                pattern_cond = torch.tensor(self.centroids[next_p], dtype=torch.float32).unsqueeze(0).to(self.device)
                for t in reversed(range(DIFFUSION_STEPS)):
                    noise = self.gen_module.diffusion.p_sample(noise, torch.tensor([t], device=self.device), pattern_cond)
                    
            next_seg = noise.squeeze().cpu().numpy()
            # Apply magnitude scaling & interpolate to duration scaling
            scaled_seg = F.interpolate(
                torch.tensor(next_seg * next_b, dtype=torch.float32).unsqueeze(0).unsqueeze(0),
                size=int(next_a * L_MAX), mode='linear'
            ).squeeze().numpy()
            
            synth.extend(scaled_seg)
            curr_p, curr_a, curr_b = torch.tensor([next_p], device=self.device), torch.tensor([next_a], device=self.device), torch.tensor([next_b], device=self.device)
            steps += 1
        return np.array(synth[:target_len])

# =============================================================================
# 5. EXECUTION PIPELINE
# =============================================================================
if __name__ == "__main__":
    # 1. Generate Synthetic Financial Data
    np.random.seed(42)
    T = 5000
    base_patterns = [np.sin(np.linspace(0, 2*np.pi, 15)), np.cos(np.linspace(0, 4*np.pi, 12)), np.linspace(-1, 1, 18)]
    X = []
    t = 0
    while t < T:
        pat = base_patterns[np.random.randint(len(base_patterns))]
        scale = np.random.uniform(0.5, 2.0)
        mag = np.random.uniform(0.8, 1.5)
        noise = np.random.normal(0, 0.1, len(pat))
        seg = pat * scale * mag + noise
        X.extend(seg)
        t += len(seg)
    X = np.array(X[:T])

    # 2. Initialize & Train FTS-Diffusion
    model = FTSDiffusion()
    model.train_sisc(X)
    model.train_generation()
    model.train_evolution()
    
    # 3. Generate & Compare
    init_seg = X[:21]
    X_synth = model.sample(init_seg, target_len=1000)
    
    print("\n" + "="*50)
    print(f"✅ Synthesized series length: {len(X_synth)}")
    print(f"📊 Real Mean/Std: {np.mean(X[:1000]):.4f} / {np.std(X[:1000]):.4f}")
    print(f"📊 Synth Mean/Std: {np.mean(X_synth):.4f} / {np.std(X_synth):.4f}")
    print("="*50)