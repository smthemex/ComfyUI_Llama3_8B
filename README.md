# ComfyUI_Llama3_8B
 Llama3_8B for comfyUI， using pipeline workflow
-----
Suport models 
----
meta-llama/Meta-Llama-3-8B-Instruct  
gradientai/Llama-3-8B-Instruct-262k  
nvidia/Llama3-ChatQA-1.5-8B   
openbmb/MiniCPM-Llama3-V-2_5   
...

Update
-----
2024-05-23 更新，加入"openbmb/MiniCPM-Llama3-V-2_5"和模型选择菜单节点  
Updated on May 23, 2024, adding "openbmb/MiniCPM-Llama3-V-2-5" and model selection menu node   

Use
----

下载模型,填写repoid，如“X:/meta-llama/Meta-Llama-3-8B-Instruct"的本地绝对路径，即可使用。   
其他不需要许可的微调模型，可以直接填写，如"gradientai/Llama-3-8B-Instruct-262k"，便直接下载模型。 

Download the model,Fill in the repoid, such as the local absolute path of "X:/meta llama/Meta Llama-3-8B Instrument", and it can be used.  
Other fine-tuning models that do not require permission can be filled in directly, such as "gradientai/Lama-3-8B-Instrument-262k", and the model can be downloaded directly. Domestic users should pay attention to downloading in advance.   

Example
----
![](https://github.com/smthemex/ComfyUI_Llama3_8B/blob/main/example/example1.png)

![](https://github.com/smthemex/ComfyUI_Llama3_8B/blob/main/example/example2.png)

![](https://github.com/smthemex/ComfyUI_Llama3_8B/blob/main/example/example3.png)

![](https://github.com/smthemex/ComfyUI_Llama3_8B/blob/main/example/example4.png)

My ComfyUI node list：
-----

1、ParlerTTS node:[ComfyUI_ParlerTTS](https://github.com/smthemex/ComfyUI_ParlerTTS)     

2、Llama3_8B node:[ComfyUI_Llama3_8B](https://github.com/smthemex/ComfyUI_Llama3_8B)      

3、HiDiffusion node：[ComfyUI_HiDiffusion_Pro](https://github.com/smthemex/ComfyUI_HiDiffusion_Pro)

4、ID_Animator node： [ComfyUI_ID_Animator](https://github.com/smthemex/ComfyUI_ID_Animator)       

5、StoryDiffusion node：[ComfyUI_StoryDiffusion](https://github.com/smthemex/ComfyUI_StoryDiffusion)  

6、Pops node：[ComfyUI_Pops](https://github.com/smthemex/ComfyUI_Pops)

7、stable-audio-open-1.0 node ：[ComfyUI_StableAudio_Open](https://github.com/smthemex/ComfyUI_StableAudio_Open)       

8、GLM4 node：[ComfyUI_ChatGLM_API](https://github.com/smthemex/ComfyUI_ChatGLM_API)

9、CustomNet node：[ComfyUI_CustomNet](https://github.com/smthemex/ComfyUI_CustomNet)         

10、Pipeline_Tool node :[ComfyUI_Pipeline_Tool](https://github.com/smthemex/ComfyUI_Pipeline_Tool)    

11、Pic2Story node :[ComfyUI_Pic2Story](https://github.com/smthemex/ComfyUI_Pic2Story)

12、PBR_Maker node:[ComfyUI_PBR_Maker](https://github.com/smthemex/ComfyUI_PBR_Maker) 

Citation
------

``` python  
@article{yu2023rlhf,
  title={Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback},
  author={Yu, Tianyu and Yao, Yuan and Zhang, Haoye and He, Taiwen and Han, Yifeng and Cui, Ganqu and Hu, Jinyi and Liu, Zhiyuan and Zheng, Hai-Tao and Sun, Maosong and others},
  journal={arXiv preprint arXiv:2312.00849},
  year={2023}
}
@article{viscpm,
    title={Large Multilingual Models Pivot Zero-Shot Multimodal Learning across Languages}, 
    author={Jinyi Hu and Yuan Yao and Chongyi Wang and Shan Wang and Yinxu Pan and Qianyu Chen and Tianyu Yu and Hanghao Wu and Yue Zhao and Haoye Zhang and Xu Han and Yankai Lin and Jiao Xue and Dahai Li and Zhiyuan Liu and Maosong Sun},
    journal={arXiv preprint arXiv:2308.12038},
    year={2023}
}
@article{xu2024llava-uhd,
  title={{LLaVA-UHD}: an LMM Perceiving Any Aspect Ratio and High-Resolution Images},
  author={Xu, Ruyi and Yao, Yuan and Guo, Zonghao and Cui, Junbo and Ni, Zanlin and Ge, Chunjiang and Chua, Tat-Seng and Liu, Zhiyuan and Huang, Gao},
  journal={arXiv preprint arXiv:2403.11703},
  year={2024}
}
```
``` python 
@article{liu2024chatqa,
  title={ChatQA: Surpassing GPT-4 on Conversational QA and RAG},
  author={Liu, Zihan and Ping, Wei and Roy, Rajarshi and Xu, Peng and Lee, Chankyu and Shoeybi, Mohammad and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:2401.10225},
  year={2024}}
```
``` python 
@article{llama3modelcard,
title={Llama 3 Model Card},
author={AI@Meta},
year={2024},
url = {https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md}
```

