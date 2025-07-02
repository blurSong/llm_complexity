from llm_complexity import auto_model, calc_inference_complexity, how_many_experts

shrink_configs = {
    "deepseek-ai/DeepSeek-V3-0324": {"num_hidden_layers": 6, "first_k_dense_replace": 1},
    "mlx-community/Meta-Llama-3.1-405B-4bit": {"num_hidden_layers": 16},  # 126
    "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit": {"num_hidden_layers": 24},  # 48
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit": {"num_hidden_layers": 40},  # 80
}

hf_repos = [
    "mlx-community/Meta-Llama-3.1-405B-4bit",
    "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
    "deepseek-ai/DeepSeek-V3-0324",
    "mlx-community/Meta-Llama-3.1-70B-Instruct-4bit",
]


def test_run():
    hf_repo = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
    shrink = False
    p, g, b = 4096, 128, 1
    axwy = "a16w4"

    shrink_config = shrink_configs[hf_repo] if shrink else None
    model = auto_model(hf_repo, "models", shrink_config)
    calc_inference_complexity(model, prompt=p, output=g, batch=b, axwy=axwy, verbose=False)


def test_hme():
    print(how_many_experts(256, 8, 8))
    print(how_many_experts(256, 16, 8))
    print(how_many_experts(256, 32, 8))
    print(how_many_experts(256, 1024, 8))


if __name__ == "__main__":
    test_run()
