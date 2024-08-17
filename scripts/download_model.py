from transformers import AutoTokenizer, AutoModelForCausalLM

def download_model(model_name="microsoft/Phi-3-mini-4k-instruct"):
    if model_name == "microsoft/Phi-3-mini-4k-instruct": # GPU
        print("Downloading Phi-3-mini-4k-instruct model")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.save_pretrained(f"./models/{model_name}")
        tokenizer.save_pretrained(f"./models/{model_name}")
    elif model_name == "microsoft/Phi-3-mini-4k-instruct-onnx": # CPU
        from huggingface_hub import snapshot_download
        # https://huggingface.co/docs/huggingface_hub/package_reference/file_download
        
        snapshot_download(
            repo_id=model_name, allow_patterns="cpu_and_mobile/*",
            local_dir=f"./models/{model_name}",
        )


if __name__ == "__main__":
    download_model("microsoft/Phi-3-mini-4k-instruct-onnx")
