import os
import transformers
import torch

class Meta_Llama3_8B:

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING",
                               {"default": "F:/test/ComfyUI/models/diffusers/meta-llama/Meta-Llama-3-8B-Instruct"}),
                "max_new_tokens": ("INT", {"default": 256, "min": 64, "max": 4096, "step": 64, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "get_model_online": ("BOOLEAN", {"default": True},),
                "reply_language": (["", "chinese", "russian", "german", "french", "spanish", "japanese"],),
                "system_content": (
                    "STRING", {"multiline": True, "default": "你叫何小喵，是一位回复私人对话的二次元白发傲娇猫娘助手"}),
                "user_content": ("STRING", {"multiline": True, "default": "何小喵，你喜欢吃什么？"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "meta_llama3_8b"
    CATEGORY = "Meta_Llama3"

    def meta_llama3_8b(self, model_path, max_new_tokens, temperature, top_p, get_model_online, reply_language,
                       system_content, user_content):
        if reply_language is not None:
            if reply_language == "chinese":
                user_content = "".join([user_content, "用中文回复我"])
            if reply_language == "russian":
                user_content = "".join([user_content, "Ответь мне по - русски"])
            if reply_language == "german":
                user_content = "".join([user_content, "Antworte mir auf Deutsch"])
            if reply_language == "french":
                user_content = "".join([user_content, "Répondez - moi en français"])
            if reply_language == "spanish":
                user_content = "".join([user_content, "Contáctame en español"])
            if reply_language == "japanese":
                user_content = "".join([user_content, "日本語で返事して"])
        if not model_path:
            raise ValueError("need a model_path or repo_id")
        else:
            if not get_model_online:
                os.environ['TRANSFORMERS_OFFLINE'] = "1"
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            try:
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device=device,
                    device_map="auto",
                )
                messages = [
                    {"role": "system", "content": f"{system_content}"},
                    {"role": "user", "content": f"{user_content}"},
                ]

                prompt = pipeline.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                terminators = [
                    pipeline.tokenizer.eos_token_id,
                    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = pipeline(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                )
                prompt_output = (outputs[0]["generated_text"])
                text_assistant = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
                prompt_output = prompt_output.split(text_assistant, 1)[1]
                prompt_output = prompt_output.replace('*', ' *')
                # print(type(prompt_output), prompt_output)
                return (prompt_output,)
            except Exception as e:
                print(e)


NODE_CLASS_MAPPINGS = {
    "Meta_Llama3_8B": Meta_Llama3_8B
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Meta_Llama3_8B": "Meta_Llama3_8B"
}
