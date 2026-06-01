#include <torch/torch.h>

#ifdef USE_CUDA
#include <c10/cuda/CUDAFunctions.h>
#endif

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using torch::Tensor;
using torch::indexing::Slice;
namespace F = torch::nn::functional;

constexpr double kPi = 3.14159265358979323846;

struct Args {
    std::string batch_path = "cpp_experiments/batches/batch.bin";
    std::string device = "cpu";
    int64_t vocab_size = 90;
    int64_t lstm_size = 400;
    int64_t K = 10;
    int64_t n_mixtures = 20;
    int64_t warmup = 1;
    int64_t iters = 5;
    double kappa_initial_bias = -4.0;
    double lr = 0.00003;
    double attention_loss_weight = 0.01;
    double grad_clip = 10.0;
    double max_pen_up_pos_weight = 100.0;
};

std::string next_arg(int& index, int argc, char** argv) {
    if (index + 1 >= argc) {
        throw std::runtime_error(std::string("Missing value for ") + argv[index]);
    }
    return argv[++index];
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--batch") {
            args.batch_path = next_arg(i, argc, argv);
        } else if (key == "--device") {
            args.device = next_arg(i, argc, argv);
        } else if (key == "--vocab-size") {
            args.vocab_size = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--lstm-size") {
            args.lstm_size = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--K") {
            args.K = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--n-mixtures") {
            args.n_mixtures = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--warmup") {
            args.warmup = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--iters") {
            args.iters = std::stoll(next_arg(i, argc, argv));
        } else if (key == "--kappa-initial-bias") {
            args.kappa_initial_bias = std::stod(next_arg(i, argc, argv));
        } else if (key == "--lr") {
            args.lr = std::stod(next_arg(i, argc, argv));
        } else if (key == "--attention-loss-weight") {
            args.attention_loss_weight = std::stod(next_arg(i, argc, argv));
        } else if (key == "--grad-clip") {
            args.grad_clip = std::stod(next_arg(i, argc, argv));
        } else if (key == "--max-pen-up-pos-weight") {
            args.max_pen_up_pos_weight = std::stod(next_arg(i, argc, argv));
        } else {
            throw std::runtime_error("Unknown argument: " + key);
        }
    }
    return args;
}

torch::Device make_device(const std::string& requested) {
    if (requested == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (requested == "cuda") {
#ifdef USE_CUDA
        if (!torch::cuda::is_available()) {
            throw std::runtime_error("CUDA was requested, but torch::cuda::is_available() is false.");
        }
        return torch::Device(torch::kCUDA, 0);
#else
        throw std::runtime_error("CUDA was requested, but this LibTorch build has no CUDA support.");
#endif
    }
    throw std::runtime_error("Unsupported device: " + requested);
}

template <typename T>
T read_value(std::ifstream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) {
        throw std::runtime_error("Unexpected end of batch file.");
    }
    return value;
}

Tensor read_tensor(std::ifstream& input) {
    int32_t dtype_code = read_value<int32_t>(input);
    int32_t ndim = read_value<int32_t>(input);
    if (ndim <= 0 || ndim > 8) {
        throw std::runtime_error("Invalid tensor ndim in batch file.");
    }

    std::vector<int64_t> sizes;
    sizes.reserve(ndim);
    int64_t numel = 1;
    for (int32_t i = 0; i < ndim; ++i) {
        int64_t dim = read_value<int64_t>(input);
        sizes.push_back(dim);
        numel *= dim;
    }

    if (dtype_code == 0) {
        std::vector<float> buffer(static_cast<size_t>(numel));
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(float)));
        if (!input) {
            throw std::runtime_error("Cannot read float tensor data from batch file.");
        }
        return torch::from_blob(buffer.data(), sizes, torch::kFloat32).clone();
    }
    if (dtype_code == 1) {
        std::vector<int64_t> buffer(static_cast<size_t>(numel));
        input.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(buffer.size() * sizeof(int64_t)));
        if (!input) {
            throw std::runtime_error("Cannot read int64 tensor data from batch file.");
        }
        return torch::from_blob(buffer.data(), sizes, torch::kInt64).clone();
    }
    throw std::runtime_error("Unsupported dtype code in batch file.");
}

std::vector<Tensor> load_batch(const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open batch file: " + path);
    }

    char magic[9] = {};
    input.read(magic, 9);
    if (std::string(magic, 9) != "HWBATCH1\n") {
        throw std::runtime_error("Invalid batch file magic. Re-run prepare_batch.py.");
    }

    int32_t tensor_count = read_value<int32_t>(input);
    std::vector<Tensor> tensors;
    tensors.reserve(tensor_count);
    for (int32_t i = 0; i < tensor_count; ++i) {
        tensors.push_back(read_tensor(input));
    }
    return tensors;
}

void sync_device(const torch::Device& device) {
#ifdef USE_CUDA
    if (device.is_cuda()) {
        c10::cuda::device_synchronize();
    }
#else
    (void)device;
#endif
}

double elapsed_ms(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TextEmbeddingImpl : torch::nn::Module {
    int64_t vocab_size;
    int64_t embedding_dim;

    explicit TextEmbeddingImpl(int64_t vocab_size_)
        : vocab_size(vocab_size_), embedding_dim(vocab_size_) {}

    Tensor forward(const Tensor& text_indices) {
        return torch::one_hot(text_indices, vocab_size).to(torch::kFloat32);
    }
};
TORCH_MODULE(TextEmbedding);

struct WindowLayerImpl : torch::nn::Module {
    int64_t K;
    torch::nn::Linear linear{nullptr};

    WindowLayerImpl(int64_t hidden_size, int64_t K_, int64_t char_embed_dim, double kappa_initial_bias)
        : K(K_), linear(register_module("linear", torch::nn::Linear(hidden_size, 3 * K_))) {
        (void)char_embed_dim;
        torch::NoGradGuard no_grad;
        linear->bias.slice(0, 2 * K, 3 * K).fill_(kappa_initial_bias);
    }

    std::tuple<Tensor, Tensor> forward(const Tensor& h1, const Tensor& char_embeddings, const Tensor& kappa_prev) {
        auto batch_size = char_embeddings.size(0);
        auto U = char_embeddings.size(1);
        (void)batch_size;

        auto params = linear->forward(h1);
        auto chunks = params.split(K, -1);
        auto alpha_hat = chunks[0];
        auto beta_hat = chunks[1];
        auto kappa_hat = chunks[2];

        auto alpha = torch::exp(torch::clamp(alpha_hat, -7.0, 7.0));
        auto beta = torch::exp(torch::clamp(beta_hat, -7.0, 7.0));
        auto kappa = kappa_prev + torch::exp(torch::clamp(kappa_hat, -7.0, 7.0));

        auto options = h1.options();
        auto u = torch::arange(U, options).view({1, U, 1});
        auto diff = kappa.unsqueeze(1) - u;
        auto phi_components = alpha.unsqueeze(1) * torch::exp(-beta.unsqueeze(1) * diff.pow(2));
        auto phi = phi_components.sum(2);
        auto w = torch::bmm(phi.unsqueeze(1), char_embeddings).squeeze(1);
        return {w, kappa};
    }
};
TORCH_MODULE(WindowLayer);

struct DecoderOutput {
    Tensor e_logit;
    Tensor pi;
    Tensor mu;
    Tensor sigma;
    Tensor rho;
    Tensor h1;
    Tensor c1;
    Tensor h2;
    Tensor c2;
    Tensor h3;
    Tensor c3;
    Tensor w;
    Tensor kappa;
};

struct DecoderImpl : torch::nn::Module {
    int64_t lstm_size;
    int64_t K;
    int64_t n_mixtures;
    torch::nn::LSTMCell lstm1{nullptr};
    torch::nn::LSTMCell lstm2{nullptr};
    torch::nn::LSTMCell lstm3{nullptr};
    WindowLayer window{nullptr};
    torch::nn::Linear output_linear{nullptr};

    DecoderImpl(
        int64_t input_size,
        int64_t char_embed_dim,
        int64_t lstm_size_,
        int64_t K_,
        int64_t n_mixtures_,
        double kappa_initial_bias)
        : lstm_size(lstm_size_), K(K_), n_mixtures(n_mixtures_) {
        lstm1 = register_module("lstm1", torch::nn::LSTMCell(input_size + char_embed_dim, lstm_size));
        lstm2 = register_module("lstm2", torch::nn::LSTMCell(input_size + lstm_size + char_embed_dim, lstm_size));
        lstm3 = register_module("lstm3", torch::nn::LSTMCell(input_size + lstm_size + char_embed_dim, lstm_size));
        window = register_module("window", WindowLayer(lstm_size, K, char_embed_dim, kappa_initial_bias));
        int64_t out_dim = input_size + char_embed_dim + 3 * lstm_size;
        output_linear = register_module("output_linear", torch::nn::Linear(out_dim, 1 + n_mixtures * 6));
    }

    DecoderOutput forward(
        const Tensor& x_t,
        const Tensor& e_prev,
        const Tensor& char_embeddings,
        const Tensor& h1,
        const Tensor& c1,
        const Tensor& h2,
        const Tensor& c2,
        const Tensor& h3,
        const Tensor& c3,
        const Tensor& kappa_prev,
        const Tensor& w_prev) {
        auto inp1 = torch::cat({x_t, e_prev, w_prev}, 1);
        auto h1c1 = lstm1->forward(inp1, std::make_tuple(h1, c1));
        auto h1_next = std::get<0>(h1c1);
        auto c1_next = std::get<1>(h1c1);

        auto wk = window->forward(h1_next, char_embeddings, kappa_prev);
        auto w_t = std::get<0>(wk);
        auto kappa_t = std::get<1>(wk);

        auto inp2 = torch::cat({x_t, e_prev, h1_next, w_t}, 1);
        auto h2c2 = lstm2->forward(inp2, std::make_tuple(h2, c2));
        auto h2_next = std::get<0>(h2c2);
        auto c2_next = std::get<1>(h2c2);

        auto inp3 = torch::cat({x_t, e_prev, h2_next, w_t}, 1);
        auto h3c3 = lstm3->forward(inp3, std::make_tuple(h3, c3));
        auto h3_next = std::get<0>(h3c3);
        auto c3_next = std::get<1>(h3c3);

        auto out_h = torch::cat({x_t, e_prev, w_t, h1_next, h2_next, h3_next}, 1);
        auto y_hat = output_linear->forward(out_h);

        auto n_mix = n_mixtures;
        auto e_logit = y_hat.slice(1, 0, 1);
        auto mdn_params = y_hat.slice(1, 1);

        auto pi = torch::softmax(mdn_params.slice(1, 0, n_mix), -1);
        auto mu = mdn_params.slice(1, n_mix, 3 * n_mix).reshape({-1, n_mix, 2});
        auto sigma_hat = mdn_params.slice(1, 3 * n_mix, 5 * n_mix);
        auto sigma = torch::exp(torch::clamp(sigma_hat, -7.0, 7.0)).reshape({-1, n_mix, 2});
        auto rho = 0.999 * torch::tanh(mdn_params.slice(1, 5 * n_mix, 6 * n_mix));

        return {
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
        };
    }
};
TORCH_MODULE(Decoder);

struct ModelOutput {
    Tensor e_logit;
    Tensor pi;
    Tensor mu;
    Tensor sigma;
    Tensor rho;
    Tensor kappa;
};

struct HandwritingSynthesisImpl : torch::nn::Module {
    TextEmbedding text_embed{nullptr};
    Decoder decoder{nullptr};

    HandwritingSynthesisImpl(int64_t vocab_size, int64_t lstm_size, int64_t K, int64_t n_mixtures, double kappa_initial_bias) {
        text_embed = register_module("text_embed", TextEmbedding(vocab_size));
        decoder = register_module(
            "decoder",
            Decoder(3, vocab_size, lstm_size, K, n_mixtures, kappa_initial_bias));
    }

    ModelOutput forward(const Tensor& dxy, const Tensor& e_target, const Tensor& text_indices, const Tensor& text_lengths) {
        auto batch_size = dxy.size(0);
        auto T = dxy.size(1);
        auto device = dxy.device();

        auto char_emb = text_embed->forward(text_indices);
        auto U = char_emb.size(1);
        auto text_positions = torch::arange(U, torch::TensorOptions().device(device).dtype(torch::kLong)).unsqueeze(0);
        auto text_mask = text_positions < text_lengths.to(device).unsqueeze(1);
        char_emb = char_emb * text_mask.unsqueeze(-1).to(torch::kFloat32);

        auto lstm_size = decoder->lstm_size;
        auto K = decoder->K;
        auto options = dxy.options();
        auto h1 = torch::zeros({batch_size, lstm_size}, options);
        auto c1 = torch::zeros({batch_size, lstm_size}, options);
        auto h2 = torch::zeros({batch_size, lstm_size}, options);
        auto c2 = torch::zeros({batch_size, lstm_size}, options);
        auto h3 = torch::zeros({batch_size, lstm_size}, options);
        auto c3 = torch::zeros({batch_size, lstm_size}, options);
        auto kappa = torch::zeros({batch_size, K}, options);
        auto w = torch::zeros({batch_size, text_embed->embedding_dim}, options);

        auto prev_x = torch::zeros({batch_size, 2}, options);
        auto prev_e = torch::zeros({batch_size, 1}, options);

        std::vector<Tensor> all_e_logit;
        std::vector<Tensor> all_pi;
        std::vector<Tensor> all_mu;
        std::vector<Tensor> all_sigma;
        std::vector<Tensor> all_rho;
        std::vector<Tensor> all_kappa;
        all_e_logit.reserve(T);
        all_pi.reserve(T);
        all_mu.reserve(T);
        all_sigma.reserve(T);
        all_rho.reserve(T);
        all_kappa.reserve(T);

        for (int64_t t = 0; t < T; ++t) {
            auto out = decoder->forward(prev_x, prev_e, char_emb, h1, c1, h2, c2, h3, c3, kappa, w);
            all_e_logit.push_back(out.e_logit);
            all_pi.push_back(out.pi);
            all_mu.push_back(out.mu);
            all_sigma.push_back(out.sigma);
            all_rho.push_back(out.rho);
            all_kappa.push_back(out.kappa);

            h1 = out.h1;
            c1 = out.c1;
            h2 = out.h2;
            c2 = out.c2;
            h3 = out.h3;
            c3 = out.c3;
            w = out.w;
            kappa = out.kappa;

            prev_x = dxy.index({Slice(), t, Slice()});
            prev_e = e_target.index({Slice(), t, Slice()});
        }

        return {
            torch::stack(all_e_logit, 1),
            torch::stack(all_pi, 1),
            torch::stack(all_mu, 1),
            torch::stack(all_sigma, 1),
            torch::stack(all_rho, 1),
            torch::stack(all_kappa, 1),
        };
    }
};
TORCH_MODULE(HandwritingSynthesis);

Tensor mdn_loss(const Tensor& pi, const Tensor& mu, const Tensor& sigma, const Tensor& rho_in, const Tensor& target_xy) {
    auto target = target_xy.unsqueeze(1).expand_as(mu);

    auto mu_x = mu.index({Slice(), Slice(), 0});
    auto mu_y = mu.index({Slice(), Slice(), 1});
    auto sigma_x = torch::clamp(sigma.index({Slice(), Slice(), 0}), 1e-4, 1e4);
    auto sigma_y = torch::clamp(sigma.index({Slice(), Slice(), 1}), 1e-4, 1e4);
    auto rho = torch::clamp(rho_in, -0.999, 0.999);

    auto dx = target.index({Slice(), Slice(), 0}) - mu_x;
    auto dy = target.index({Slice(), Slice(), 1}) - mu_y;
    auto one_minus_rho2 = torch::clamp(1.0 - rho.pow(2), 1e-5);

    auto z = (dx / sigma_x).pow(2);
    z = z + (dy / sigma_y).pow(2);
    z = z - 2.0 * rho * dx * dy / (sigma_x * sigma_y);
    z = z / one_minus_rho2;

    auto log_norm = -0.5 * z;
    log_norm = log_norm - torch::log(2.0 * kPi * sigma_x * sigma_y * torch::sqrt(one_minus_rho2));

    auto log_prob = torch::logsumexp(torch::log(torch::clamp(pi, 1e-8)) + log_norm, 1);
    return (-log_prob).mean();
}

Tensor pen_up_pos_weight(const Tensor& e_target, double max_weight) {
    auto positives = e_target.sum().clamp_min(1.0);
    auto total = torch::full({}, static_cast<double>(e_target.numel()), e_target.options());
    auto negatives = total - positives;
    return (negatives / positives).clamp(1.0, max_weight).detach().reshape({1});
}

Tensor kappa_progress_loss(const Tensor& kappa, const Tensor& lengths, const Tensor& text_lengths) {
    auto B = kappa.size(0);
    auto T = kappa.size(1);
    (void)B;
    auto device = kappa.device();
    auto dtype = kappa.dtype();

    auto positions = torch::arange(T, torch::TensorOptions().device(device).dtype(torch::kLong)).unsqueeze(0);
    auto mask = positions < lengths.to(device).unsqueeze(1);

    auto time = torch::arange(1, T + 1, torch::TensorOptions().device(device).dtype(dtype)).unsqueeze(0);
    auto target = time / lengths.to(device).to(dtype).clamp_min(1).unsqueeze(1);
    target = target * text_lengths.to(device).to(dtype).clamp_min(1).unsqueeze(1);

    auto pred = kappa.mean(2);
    auto pred_masked = pred.masked_select(mask);
    auto target_masked = target.masked_select(mask);
    return F::smooth_l1_loss(pred_masked, target_masked);
}

struct Losses {
    Tensor loss;
    Tensor mdn;
    Tensor pen;
    Tensor attn;
};

Losses batch_losses(
    const ModelOutput& outputs,
    const Tensor& dxy,
    const Tensor& e_target,
    const Tensor& lengths,
    const Tensor& text_lengths,
    double attention_loss_weight,
    double max_pen_up_pos_weight) {
    auto B = dxy.size(0);
    auto T = dxy.size(1);
    auto device = dxy.device();
    auto positions = torch::arange(T, torch::TensorOptions().device(device).dtype(torch::kLong)).unsqueeze(0);
    auto mask = positions < lengths.to(device).unsqueeze(1);
    auto flat_mask = mask.reshape({-1});

    auto pi = outputs.pi.reshape({B * T, -1}).index({flat_mask});
    auto mu = outputs.mu.reshape({B * T, outputs.mu.size(2), 2}).index({flat_mask});
    auto sigma = outputs.sigma.reshape({B * T, outputs.sigma.size(2), 2}).index({flat_mask});
    auto rho = outputs.rho.reshape({B * T, -1}).index({flat_mask});
    auto target = dxy.reshape({B * T, 2}).index({flat_mask});
    auto loss_mdn = mdn_loss(pi, mu, sigma, rho, target);

    auto e_logits = outputs.e_logit.reshape({B * T, 1}).index({flat_mask});
    auto e_targets = e_target.reshape({B * T, 1}).index({flat_mask});
    auto pos_weight = pen_up_pos_weight(e_targets, max_pen_up_pos_weight);
    auto bce_options = F::BinaryCrossEntropyWithLogitsFuncOptions().pos_weight(pos_weight);
    auto loss_e = F::binary_cross_entropy_with_logits(e_logits, e_targets, bce_options);

    auto loss_attn = kappa_progress_loss(outputs.kappa, lengths, text_lengths);
    auto loss = loss_mdn + loss_e + attention_loss_weight * loss_attn;
    return {loss, loss_mdn.detach(), loss_e.detach(), loss_attn.detach()};
}

struct StepTiming {
    double forward_ms = 0.0;
    double loss_ms = 0.0;
    double backward_ms = 0.0;
    double step_ms = 0.0;
    double total_ms = 0.0;
    double loss = 0.0;
};

StepTiming run_step(
    HandwritingSynthesis& model,
    torch::optim::RMSprop& optimizer,
    const Tensor& dxy,
    const Tensor& e_target,
    const Tensor& text,
    const Tensor& text_lengths,
    const Tensor& lengths,
    const Args& args,
    const torch::Device& device) {
    model->train();
    sync_device(device);
    auto t0 = std::chrono::steady_clock::now();

    optimizer.zero_grad();
    auto outputs = model->forward(dxy, e_target, text, text_lengths);
    sync_device(device);
    auto t1 = std::chrono::steady_clock::now();

    auto losses = batch_losses(
        outputs,
        dxy,
        e_target,
        lengths,
        text_lengths,
        args.attention_loss_weight,
        args.max_pen_up_pos_weight);
    sync_device(device);
    auto t2 = std::chrono::steady_clock::now();

    losses.loss.backward();
    sync_device(device);
    auto t3 = std::chrono::steady_clock::now();

    if (args.grad_clip > 0.0) {
        torch::nn::utils::clip_grad_norm_(model->parameters(), args.grad_clip);
    }
    optimizer.step();
    sync_device(device);
    auto t4 = std::chrono::steady_clock::now();

    return {
        elapsed_ms(t0, t1),
        elapsed_ms(t1, t2),
        elapsed_ms(t2, t3),
        elapsed_ms(t3, t4),
        elapsed_ms(t0, t4),
        losses.loss.detach().cpu().item<double>(),
    };
}

int main(int argc, char** argv) {
    try {
        auto args = parse_args(argc, argv);
        auto device = make_device(args.device);
        torch::manual_seed(20260601);

        std::vector<Tensor> batch = load_batch(args.batch_path);
        if (batch.size() != 5) {
            throw std::runtime_error("Expected batch.pt to contain [dxy, e, text, text_lengths, length].");
        }

        auto dxy = batch[0].to(device);
        auto e_target = batch[1].to(device);
        auto text = batch[2].to(device, torch::kLong);
        auto text_lengths = batch[3].to(device, torch::kLong);
        auto lengths = batch[4].to(device, torch::kLong);

        HandwritingSynthesis model(
            args.vocab_size,
            args.lstm_size,
            args.K,
            args.n_mixtures,
            args.kappa_initial_bias);
        model->to(device);

        torch::optim::RMSprop optimizer(
            model->parameters(),
            torch::optim::RMSpropOptions(args.lr).alpha(0.95).momentum(0.9).eps(1e-4));

        std::cout << "C++ train-step benchmark\n";
        std::cout << "device: " << device << "\n";
        std::cout << "batch: " << args.batch_path << "\n";
        std::cout << "shape: B=" << dxy.size(0) << " T=" << dxy.size(1)
                  << " U=" << text.size(1) << "\n";
        std::cout << "model: vocab=" << args.vocab_size << " lstm_size=" << args.lstm_size
                  << " K=" << args.K << " n_mixtures=" << args.n_mixtures << "\n";
        std::cout << "warmup: " << args.warmup << " measured: " << args.iters << "\n\n";

        for (int64_t i = 0; i < args.warmup; ++i) {
            auto timing = run_step(model, optimizer, dxy, e_target, text, text_lengths, lengths, args, device);
            std::cout << "warmup " << (i + 1) << ": total=" << std::fixed << std::setprecision(1)
                      << timing.total_ms << " ms loss=" << std::setprecision(4) << timing.loss << "\n";
        }

        StepTiming total;
        for (int64_t i = 0; i < args.iters; ++i) {
            auto timing = run_step(model, optimizer, dxy, e_target, text, text_lengths, lengths, args, device);
            total.forward_ms += timing.forward_ms;
            total.loss_ms += timing.loss_ms;
            total.backward_ms += timing.backward_ms;
            total.step_ms += timing.step_ms;
            total.total_ms += timing.total_ms;
            total.loss = timing.loss;
            std::cout << "iter " << std::setw(2) << (i + 1)
                      << ": forward=" << std::fixed << std::setprecision(1) << timing.forward_ms
                      << "ms loss=" << timing.loss_ms
                      << "ms backward=" << timing.backward_ms
                      << "ms step=" << timing.step_ms
                      << "ms total=" << timing.total_ms
                      << "ms train_loss=" << std::setprecision(4) << timing.loss << "\n";
        }

        auto denom = std::max<int64_t>(args.iters, 1);
        std::cout << "\nTiming summary\n";
        std::cout << "forward_ms_avg: " << std::fixed << std::setprecision(1) << total.forward_ms / denom << "\n";
        std::cout << "loss_ms_avg: " << total.loss_ms / denom << "\n";
        std::cout << "backward_ms_avg: " << total.backward_ms / denom << "\n";
        std::cout << "step_ms_avg: " << total.step_ms / denom << "\n";
        std::cout << "total_ms_avg: " << total.total_ms / denom << "\n";
        std::cout << "last_loss: " << std::setprecision(4) << total.loss << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 1;
    }
}
