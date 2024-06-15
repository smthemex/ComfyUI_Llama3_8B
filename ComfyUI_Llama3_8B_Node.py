import os
import sys
import re
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
from PIL import Image
import folder_paths

dir_path = os.path.dirname(os.path.abspath(__file__))
path_dir = os.path.dirname(dir_path)
file_path = os.path.dirname(path_dir)
def string_punctuation_bool(string_in):
    pattern = r"[^\w\s]$"
    string_bool = bool(re.search(pattern, string_in))
    return string_bool

def trans_reply(reply_language,user_content):
    if string_punctuation_bool(user_content):
        join_punctuation = " "
    else:
        join_punctuation = ","
    if reply_language == "chinese":
        user_content = f"{join_punctuation}".join([user_content, "用中文回复我"])
    elif reply_language == "russian":
        user_content = f"{join_punctuation}".join([user_content, "Ответь мне по - русски"])
    elif reply_language == "german":
        user_content = f"{join_punctuation}".join([user_content, "Antworte mir auf Deutsch"])
    elif reply_language == "french":
        user_content = f"{join_punctuation}".join([user_content, "Répondez - moi en français"])
    elif reply_language == "spanish":
        user_content = f"{join_punctuation}".join([user_content, "Contáctame en español"])
    elif reply_language == "japanese":
        user_content = f"{join_punctuation}".join([user_content, "日本語で返事して"])
    elif reply_language == "english":
        user_content = f"{join_punctuation}".join([user_content, "answer me in English"])
    else:
        user_content = f"{join_punctuation}".join([user_content, "Reply to me in the language of my question mentioned above"])
    return user_content

paths = []
for search_path in folder_paths.get_folder_paths("diffusers"):
    if os.path.exists(search_path):
        for root, subdir, files in os.walk(search_path, followlinks=True):
            if "model.safetensors.index.json" in files:
                paths.append(os.path.relpath(root, start=search_path))

if paths != []:
    paths = [] + [x for x in paths if x]
else:
    paths = ["no llama3 model in default diffusers directory", ]


def get_local_path(file_path, model_path):
    path = os.path.join(file_path, "models", "diffusers", model_path)
    model_path = os.path.normpath(path)
    if sys.platform=='win32':
        model_path = model_path.replace('\\', "/")
    return model_path


def tensor_to_image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def get_instance_path(path):
    instance_path = os.path.normpath(path)
    if sys.platform=='win32':
        instance_path = instance_path.replace('\\', "/")
    return instance_path


class Local_Or_Repo_Choice:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "local_model_path": (paths,),
                "repo_id": ("STRING", {"default": "THUDM/cogvlm2-llama3-chat-19B"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("repo_id",)
    FUNCTION = "repo_choice"
    CATEGORY = "Meta_Llama3"

    def repo_choice(self, local_model_path, repo_id):
        if repo_id == "":
            if local_model_path == ["no llama3 model in default diffusers directory", ]:
                raise "you need fill repo_id or download model in diffusers directory "
            elif local_model_path != ["no llama3 model in default diffusers directory", ]:
                model_path = get_local_path(file_path, local_model_path)
                repo_id = get_instance_path(model_path)
        elif repo_id != "" and repo_id.find("/") == -1:
            raise "Incorrect repo_id format"
        elif repo_id != "" and repo_id.find("\\") != -1:
            repo_id = get_instance_path(repo_id)
        return (repo_id,)


class Meta_Llama3_8B:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 256, "min": 32, "max": 4096, "step": 32, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "get_model_online": ("BOOLEAN", {"default": True},),
                "reply_language": (["english", "chinese", "russian", "german", "french", "spanish", "japanese","Original_language"],),
                "system_content": (
                    "STRING", {"multiline": True, "default": "你叫何小喵，是一位回复私人对话的二次元白发傲娇猫娘助手"}),
                "user_content": ("STRING", {"multiline": True, "default": "何小喵，你喜欢吃什么？"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "meta_llama3_8b"
    CATEGORY = "Meta_Llama3"

    def meta_llama3_8b(self, repo_id, max_new_tokens, temperature, top_p, get_model_online, reply_language,
                       system_content, user_content):
        user_content = trans_reply(reply_language, user_content)
        if not get_model_online:
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
        try:
            pipeline = transformers.pipeline(
                "text-generation",
                model=repo_id,
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
            return (e,)


class ChatQA_1p5_8b:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "repo_id": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 128, "min": 32, "max": 4096, "step": 32, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.6, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "get_model_online": ("BOOLEAN", {"default": True},),
                "reply_language": (["english", "chinese", "russian", "german", "french", "spanish", "japanese","Original_language"],),
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

    def chatqa_1p5_8b(self, repo_id, max_new_tokens, temperature, top_p, get_model_online, reply_language, system,
                      instruction,
                      user_content):
        user_content = trans_reply(reply_language, user_content)
        if not get_model_online:
            os.environ['TRANSFORMERS_OFFLINE'] = "1"
        try:
            tokenizer = AutoTokenizer.from_pretrained(repo_id)
            model = AutoModelForCausalLM.from_pretrained(repo_id, torch_dtype=torch.float16, device_map="auto")
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
            return (e,)


class MiniCPM_Llama3_V25:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "repo_id": ("STRING", {"forceInput": True}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 32, "max": 4096, "step": 32, "display": "number"}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "top_p": (
                    "FLOAT",
                    {"default": 0.9, "min": 0.01, "max": 0.99, "step": 0.01, "round": False, "display": "number"}),
                "reply_language": (["english", "chinese", "russian", "german", "french", "spanish", "japanese","Original_language"],),
                "question": ("STRING", {"multiline": True,
                                        "default": "What is in the image?"})
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "minicpm_llama3_v25"
    CATEGORY = "Meta_Llama3"

    def minicpm_llama3_v25(self, image, repo_id, max_new_tokens, temperature, top_p, reply_language,
                           question):
        question = trans_reply(reply_language, question)
        try:
            model = AutoModel.from_pretrained(repo_id, trust_remote_code=True,
                                              torch_dtype=torch.float16)
            model = model.to(device='cuda')
            tokenizer = AutoTokenizer.from_pretrained(repo_id, trust_remote_code=True)
            model.eval()
            image = tensor_to_image(image)
            msgs = [{'role': 'user', 'content': question}]
            res = model.chat(
                image=image,
                msgs=msgs,
                max_new_tokens=max_new_tokens,
                tokenizer=tokenizer,
                sampling=True,
                top_p=top_p,
                temperature=temperature
            )
            # print(res)
            return (res,)
        except Exception as e:
            return (e,)


NODE_CLASS_MAPPINGS = {
    "Local_Or_Repo_Choice": Local_Or_Repo_Choice,
    "Meta_Llama3_8B": Meta_Llama3_8B,
    "ChatQA_1p5_8b": ChatQA_1p5_8b,
    "MiniCPM_Llama3_V25": MiniCPM_Llama3_V25
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Local_Or_Repo_Choice": "Local_Or_Repo_Choice",
    "Meta_Llama3_8B": "Meta_Llama3_8B",
    "ChatQA_1p5_8b": "ChatQA_1p5_8b",
    "MiniCPM_Llama3_V25": "MiniCPM_Llama3_V25"
}
