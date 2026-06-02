import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = vocab_size

    def forward(self, text_indices):
        return F.one_hot(text_indices, num_classes=self.vocab_size).float()


class WindowLayer(nn.Module):
    def __init__(self, hidden_size, K, char_embed_dim, kappa_initial_bias=-4.0):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(hidden_size, 3 * K)
        with torch.no_grad():
            self.linear.bias[2 * K : 3 * K].fill_(kappa_initial_bias)

    def forward(self, h1, char_embeddings, kappa_prev):
        batch_size, U, embed_dim = char_embeddings.shape
        params = self.linear(h1)
        alpha_hat, beta_hat, kappa_hat = params.split(self.K, dim=-1)

        alpha = torch.exp(torch.clamp(alpha_hat, -7.0, 7.0))
        beta = torch.exp(torch.clamp(beta_hat, -7.0, 7.0))
        kappa = kappa_prev + torch.exp(torch.clamp(kappa_hat, -7.0, 7.0))

        u = torch.arange(U, device=h1.device, dtype=h1.dtype).view(1, U, 1)
        phi_components = (
            alpha.unsqueeze(1)
            * torch.exp(-beta.unsqueeze(1) * (kappa.unsqueeze(1) - u) ** 2)
        )
        phi = phi_components.sum(dim=2)

        w = torch.bmm(phi.unsqueeze(1), char_embeddings).squeeze(1)
        return w, kappa, phi


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        char_embed_dim,
        lstm_size,
        K,
        n_mixtures,
        kappa_initial_bias=-4.0,
    ):
        super().__init__()
        self.lstm_size = lstm_size
        self.K = K
        self.n_mixtures = n_mixtures

        self.lstm1 = nn.LSTMCell(input_size + char_embed_dim, lstm_size)
        self.lstm2 = nn.LSTMCell(input_size + lstm_size + char_embed_dim, lstm_size)
        self.lstm3 = nn.LSTMCell(input_size + lstm_size + char_embed_dim, lstm_size)
        self.window = WindowLayer(lstm_size, K, char_embed_dim, kappa_initial_bias)

        out_dim = input_size + char_embed_dim + 3 * lstm_size
        self.output_linear = nn.Linear(out_dim, 1 + n_mixtures * 6)

    def forward(
        self,
        x_t,
        e_prev,
        char_embeddings,
        h1,
        c1,
        h2,
        c2,
        h3,
        c3,
        kappa_prev,
        w_prev,
    ):
        inp1 = torch.cat([x_t, e_prev, w_prev], dim=1)
        h1_next, c1_next = self.lstm1(inp1, (h1, c1))

        w_t, kappa_t, _ = self.window(h1_next, char_embeddings, kappa_prev)

        inp2 = torch.cat([x_t, e_prev, h1_next, w_t], dim=1)
        h2_next, c2_next = self.lstm2(inp2, (h2, c2))

        inp3 = torch.cat([x_t, e_prev, h2_next, w_t], dim=1)
        h3_next, c3_next = self.lstm3(inp3, (h3, c3))

        out_h = torch.cat([x_t, e_prev, w_t, h1_next, h2_next, h3_next], dim=1)
        y_hat = self.output_linear(out_h)

        n_mix = self.n_mixtures
        e_logit = y_hat[:, 0:1]
        mdn_params = y_hat[:, 1:]

        pi = F.softmax(mdn_params[:, :n_mix], dim=-1)
        mu = mdn_params[:, n_mix : 3 * n_mix].reshape(-1, n_mix, 2)
        sigma_hat = mdn_params[:, 3 * n_mix : 5 * n_mix]
        sigma = torch.exp(torch.clamp(sigma_hat, -7.0, 7.0)).reshape(-1, n_mix, 2)
        rho = 0.999 * torch.tanh(mdn_params[:, 5 * n_mix : 6 * n_mix])

        return (
            e_logit,
            pi,
            mu,
            sigma,
            rho,
            h1_next,
            c1_next,
            h2_next,
            c2_next,
            h3_next,
            c3_next,
            w_t,
            kappa_t,
        )


class HandwritingSynthesis(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        lstm_size,
        num_layers,
        K,
        n_mixtures,
        kappa_initial_bias=-4.0,
    ):
        super().__init__()
        self.text_embed = TextEmbedding(vocab_size, embed_dim)
        self.decoder = Decoder(
            input_size=3,
            char_embed_dim=embed_dim,
            lstm_size=lstm_size,
            K=K,
            n_mixtures=n_mixtures,
            kappa_initial_bias=kappa_initial_bias,
        )

    def _predicted_input(self, e_logit, pi, mu, sigma, rho, mode):
        with torch.no_grad():
            if mode == "mean":
                pred_x = (pi.unsqueeze(-1) * mu).sum(dim=1)
            elif mode == "sample":
                idx = torch.multinomial(pi, 1)
                idx_expanded = idx.unsqueeze(-1).expand(-1, 1, 2)
                mean = torch.gather(mu, 1, idx_expanded).squeeze(1)
                std = torch.gather(sigma, 1, idx_expanded).squeeze(1)
                corr = torch.gather(rho, 1, idx).squeeze(1)
                eps = torch.randn_like(mean)
                pred_x = torch.stack(
                    [
                        mean[:, 0] + std[:, 0] * eps[:, 0],
                        mean[:, 1]
                        + std[:, 1]
                        * (
                            corr * eps[:, 0]
                            + torch.sqrt(torch.clamp(1.0 - corr**2, min=1e-5)) * eps[:, 1]
                        ),
                    ],
                    dim=1,
                )
            else:
                mixture_idx = pi.argmax(dim=1)
                gather_idx = mixture_idx.view(-1, 1, 1).expand(-1, 1, 2)
                pred_x = torch.gather(mu, 1, gather_idx).squeeze(1)

            pred_e = (torch.sigmoid(e_logit) > 0.5).to(e_logit.dtype)
        return pred_x.detach(), pred_e.detach()

    def forward(
        self,
        dxy,
        e_target,
        text_indices,
        text_lengths,
        teacher_forcing_ratio=1.0,
        scheduled_sampling_mode="argmax",
    ):
        batch_size, T = dxy.shape[:2]
        device = dxy.device
        char_emb = self.text_embed(text_indices)
        if text_lengths is not None:
            U = char_emb.shape[1]
            text_mask = torch.arange(U, device=device).unsqueeze(0) < text_lengths.to(device).unsqueeze(1)
            char_emb = char_emb * text_mask.unsqueeze(-1).float()

        lstm_size = self.decoder.lstm_size
        h1 = torch.zeros(batch_size, lstm_size, device=device)
        c1 = torch.zeros(batch_size, lstm_size, device=device)
        h2 = torch.zeros(batch_size, lstm_size, device=device)
        c2 = torch.zeros(batch_size, lstm_size, device=device)
        h3 = torch.zeros(batch_size, lstm_size, device=device)
        c3 = torch.zeros(batch_size, lstm_size, device=device)
        kappa = torch.zeros(batch_size, self.decoder.K, device=device)
        w = torch.zeros(batch_size, self.text_embed.embedding_dim, device=device)

        prev_x = torch.zeros(batch_size, 2, device=device)
        prev_e = torch.zeros(batch_size, 1, device=device)

        all_e_logit, all_pi, all_mu, all_sigma, all_rho, all_kappa = [], [], [], [], [], []

        for t in range(T):
            (
                e_logit,
                pi,
                mu,
                sigma,
                rho,
                h1,
                c1,
                h2,
                c2,
                h3,
                c3,
                w,
                kappa,
            ) = self.decoder(
                x_t=prev_x,
                e_prev=prev_e,
                char_embeddings=char_emb,
                h1=h1,
                c1=c1,
                h2=h2,
                c2=c2,
                h3=h3,
                c3=c3,
                kappa_prev=kappa,
                w_prev=w,
            )

            all_e_logit.append(e_logit)
            all_pi.append(pi)
            all_mu.append(mu)
            all_sigma.append(sigma)
            all_rho.append(rho)
            all_kappa.append(kappa)

            if self.training and teacher_forcing_ratio < 1.0:
                pred_x, pred_e = self._predicted_input(
                    e_logit,
                    pi,
                    mu,
                    sigma,
                    rho,
                    scheduled_sampling_mode,
                )
                if teacher_forcing_ratio <= 0.0:
                    prev_x = pred_x
                    prev_e = pred_e
                else:
                    use_teacher = (
                        torch.rand(batch_size, 1, device=device) < teacher_forcing_ratio
                    )
                    prev_x = torch.where(use_teacher, dxy[:, t, :], pred_x)
                    prev_e = torch.where(use_teacher, e_target[:, t, :], pred_e)
            else:
                prev_x = dxy[:, t, :]
                prev_e = e_target[:, t, :]

        return {
            "e_logit": torch.stack(all_e_logit, dim=1),
            "pi": torch.stack(all_pi, dim=1),
            "mu": torch.stack(all_mu, dim=1),
            "sigma": torch.stack(all_sigma, dim=1),
            "rho": torch.stack(all_rho, dim=1),
            "kappa": torch.stack(all_kappa, dim=1),
        }
