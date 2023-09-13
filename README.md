# LLaMA Docker Playground

This project is compatible with LLaMA2, but you can visit the project below to experience various ways to talk to LLaMA2 **(private deployment)**: [soulteary/docker-llama2-chat](https://github.com/soulteary/docker-llama2-chat)

<img src="./assets/llama.jpg" width="50%" >

[中文教程](./assets/guide-zh.md)

A "Clean and Hygienic" LLaMA Playground, Play LLaMA with 7GB (int8) 10GB (pyllama) or 20GB (official) of VRAM.

At the same time, it provides Alpaca LoRA one-click running Docker image, which can finetune 7B / 65B models.

## How to use

To use this project, we need to do **two things**:

1. the first thing is to download the model
  - (you can download the LLaMA models from anywhere)
2. and the second thing is to build the image with the docker
  - (saves time compared to downloading from Docker Hub)

### Put the Models File in Right Place

Taking the smallest model as an example, you need to place the model related files like this:

```bash
.
└── models
    ├── 65B
    │   ├── checklist.chk
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   ├── consolidated.02.pth
    │   ├── consolidated.03.pth
    │   ├── consolidated.04.pth
    │   ├── consolidated.05.pth
    │   ├── consolidated.06.pth
    │   ├── consolidated.07.pth
    │   └── params.json
    ├── 30B
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   ├── consolidated.02.pth
    │   ├── consolidated.03.pth
    │   └── params.json
    ├── 13B
    │   ├── consolidated.00.pth
    │   ├── consolidated.01.pth
    │   └── params.json
    ├── 7B
    │   ├── consolidated.00.pth
    │   └── params.json
    └── tokenizer.model
```

### Build the LLaMA Docker Playground

If you prefer to use the official authentic model, build the docker image with the following command:

```bash
docker build -t soulteary/llama:llama . -f docker/Dockerfile.llama
```

If you wish to use a model with lower memory requirements, build the docker image with the following command:

```bash
docker build -t soulteary/llama:pyllama . -f docker/Dockerfile.pyllama
```

If you wish to use a model with **the minimum memory** requirements, build the docker image with the following command:

```bash
docker build -t soulteary/llama:int8 . -f docker/Dockerfile.int8
```

If you wish to **fine-tune** a model(7B-65B) with **the minimum memory** requirements, build the docker image with the following command:

```bash
# single GPU
docker build -t soulteary/llama:alpaca-lora-finetune . -f docker/Dockerfile.lora-finetune
# multiple GPU
docker build -t soulteary/llama:alpaca-lora-65b-finetune . -f docker/Dockerfile.lora-65b-finetune
```

### Play with the LLaMA

For official model docker images (7B almost 21GB), use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -v `pwd`/models:/app/models -p 7860:7860 -it --rm soulteary/llama:llama
```

For lower memory requirements (7B almost 13GB) docker images, use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -v `pwd`/models:/llama_data -p 7860:7860 -it --rm soulteary/llama:pyllama
```

For **the minimum memory** requirements (7B almost 7.12GB) docker images, use the following command:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -e PORT=7860 -v `pwd`/models:/app/models -p 7860:7860 -it --rm soulteary/llama:int8
```

**For fine-tune**, [read this documentation](https://soulteary.com/2023/03/25/model-finetuning-on-llama-65b-large-model-using-docker-and-alpaca-lora.html).


## Credits

- [facebookresearch/llama](https://github.com/facebookresearch/llama)
- [andrewssobral's pr](https://github.com/facebookresearch/llama/pull/126/files)
- [juncongmoo/pyllama](https://github.com/juncongmoo/pyllama)
- [tloen/llama-int8](https://github.com/tloen/llama-int8)
- [tloen/alpaca-lora](https://github.com/tloen/alpaca-lora)

## License

Follow the rules of the game and be consistent with the original project.
