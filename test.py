from llm_complexity import auto_model, calc_inference_complexity, how_many_experts


def test_llms():
    hf_repos = [
        "mlx-community/Meta-Llama-3.1-405B-4bit",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "deepseek-ai/DeepSeek-V3-0324",
    ]
    for hf_repo in hf_repos:
        model = auto_model(hf_repo, "models")
        calc_inference_complexity(model, prompt=1024, output=512, batch=1, axwy="a16w4", verbose=True)


def test_shrink():
    deepseek_v3_shrink_config = {"num_hidden_layers": 6, "first_k_dense_replace": 1}
    llama3_405b_shrink_config = {"num_hidden_layers": 16}  # 126
    llama4_scout_shrink_config = {"num_hidden_layers": 24}  # 48
    hf_repos = [
        "mlx-community/Meta-Llama-3.1-405B-4bit",
        "mlx-community/Llama-4-Scout-17B-16E-Instruct-4bit",
        "deepseek-ai/DeepSeek-V3-0324",
    ]
    for hf_repo in hf_repos:
        if "DeepSeek" in hf_repo:
            model = auto_model(hf_repo, "models", deepseek_v3_shrink_config)
        elif "Llama-3.1-405B" in hf_repo:
            model = auto_model(hf_repo, "models", llama3_405b_shrink_config)
        elif "Llama-4-Scout" in hf_repo:
            model = auto_model(hf_repo, "models", llama4_scout_shrink_config)
        for b in [
            1,
        ]:
            calc_inference_complexity(model, prompt=1024, output=512, batch=b, axwy="a32w4", verbose=False)


def test_hme():
    print(how_many_experts(256, 8, 8))
    print(how_many_experts(256, 16, 8))
    print(how_many_experts(256, 32, 8))
    print(how_many_experts(256, 1024, 8))


if __name__ == "__main__":
    test_llms()
    # test_shrink()
    # test_hme()
