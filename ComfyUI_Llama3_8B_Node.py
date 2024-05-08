import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
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
                "max_new_tokens": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "get_model_online": ("BOOLEAN", {"default": True},),
                "reply_language": (["None", "chinese", "russian", "german", "french", "spanish", "japanese"],),
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
        if reply_language == "chinese":
            user_content = "".join([user_content, "用中文回复我"])
        elif reply_language == "russian":
            user_content = "".join([user_content, "Ответь мне по - русски"])
        elif reply_language == "german":
            user_content = "".join([user_content, "Antworte mir auf Deutsch"])
        elif reply_language == "french":
            user_content = "".join([user_content, "Répondez - moi en français"])
        elif reply_language == "spanish":
            user_content = "".join([user_content, "Contáctame en español"])
        elif reply_language == "japanese":
            user_content = "".join([user_content, "日本語で返事して"])
        else:
            user_content = "".join([user_content, "answer me in English"])
        if not model_path:
            raise ValueError("need a model_path or repo_id")
        else:
            if not get_model_online:
                os.environ['TRANSFORMERS_OFFLINE'] = "1"
            try:
                pipeline = transformers.pipeline(
                    "text-generation",
                    model=model_path,
                    model_kwargs={"torch_dtype": torch.bfloat16},
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


class ChatQA_1p5_8B:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_path": ("STRING",
                               {"default": "nvidia/Llama3-ChatQA-1.5-8B"}),
                "max_new_tokens": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 32, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "get_model_online": ("BOOLEAN", {"default": True},),
                "reply_language": (["None", "chinese", "russian", "german", "french", "spanish", "japanese"],),
                "system": (
                    "STRING",
                    {"multiline": True,
                     "default": "System: This is a chat between a user and an artificial intelligence"
                                " assistant. The assistant gives helpful, detailed, and polite answers "
                                "to the user's questions based on the context. The assistant should also "
                                "indicate when the answer cannot be found in the context."}),
                "instruction": (
                    "STRING",
                    {"multiline": True, "default": "Please give a full and complete answer for the question."}),
                "user_content": ("STRING", {"multiline": True,
                                            "default": "你是一位撰写提示词的高级助理，现在给我写一个关于'一只小猫，穿着宇航服，漫步在月球表面的'的提示词"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "chatqa_1p5_8b"
    CATEGORY = "Meta_Llama3"

    def get_formatted_input(self, system, instruction, messages, context):
        for item in messages:
            if item['role'] == "user":
                # only apply this instruction for the first user turn
                item['content'] = instruction + " " + item['content']
                break

        conversation = '\n\n'.join(
            ["User: " + item["content"] if item["role"] == "user" else "Assistant: " + item["content"] for
             item in messages]) + "\n\nAssistant:"
        formatted_input = system + "\n\n" + context + "\n\n" + conversation

        return formatted_input

    def chatqa_1p5_8b(self, model_path, max_new_tokens, temperature,top_p,get_model_online, reply_language, system, instruction,
                      user_content):
        if reply_language == "chinese":
            user_content = "".join([user_content, "用中文回复我"])
        elif reply_language == "russian":
            user_content = "".join([user_content, "Ответь мне по - русски"])
        elif reply_language == "german":
            user_content = "".join([user_content, "Antworte mir auf Deutsch"])
        elif reply_language == "french":
            user_content = "".join([user_content, "Répondez - moi en français"])
        elif reply_language == "spanish":
            user_content = "".join([user_content, "Contáctame en español"])
        elif reply_language == "japanese":
            user_content = "".join([user_content, "日本語で返事して"])
        else:
            user_content = "".join([user_content, "answer me in English"])
        if not model_path:
            raise ValueError("need a model_path or repo_id")
        else:
            if not get_model_online:
                os.environ['TRANSFORMERS_OFFLINE'] = "1"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
                messages = [{"role": "user", "content": user_content}]
                document = ""
                formatted_input = self.get_formatted_input(system, instruction, messages, document)
                tokenized_prompt = tokenizer(tokenizer.bos_token + formatted_input, return_tensors="pt").to(
                    model.device)

                terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                outputs = model.generate(input_ids=tokenized_prompt.input_ids,
                                         attention_mask=tokenized_prompt.attention_mask,
                                         do_sample=True,
                                         temperature=temperature,
                                         top_p=top_p,
                                         max_new_tokens=max_new_tokens,
                                         eos_token_id=terminators)
                response = outputs[0][tokenized_prompt.input_ids.shape[-1]:]
                print(tokenizer.decode(response, skip_special_tokens=True))
                prompt_output = tokenizer.decode(response, skip_special_tokens=True)
                if ":" in prompt_output:
                    prompt_output = prompt_output.split(":", 1)[1]
                prompt_output = prompt_output.strip().strip('\'"').replace("\n", " ")
                return (prompt_output,)

            except Exception as e:
                print(e)


NODE_CLASS_MAPPINGS = {
    "Meta_Llama3_8B": Meta_Llama3_8B,
    "ChatQA_1p5_8B": ChatQA_1p5_8B
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Meta_Llama3_8B": "Meta_Llama3_8B",
    "ChatQA_1p5_8B": "ChatQA_1p5_8B"
}
