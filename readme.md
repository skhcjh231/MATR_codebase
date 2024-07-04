# Online Temporal Action Localization with Memory-Augmented Transformer

### [Youngkil Song*](https://www.linkedin.com/in/youngkil-song-8936792a3/), [Dongkeun Kim*](https://dk-kim.github.io/), [Minsu Cho](https://cvlab.postech.ac.kr/~mcho/), [Suha Kwak](https://suhakwak.github.io/)

## Requirements

- Ubuntu 22.04
- Python 3.10.9
- CUDA 11.7
- Pytorch 2.0.0

## environment installation
    
    pip install -r requirements.txt

## Download datasets & trained weights (THUMOS14 dataset only)

    link: https://drive.google.com/drive/folders/1-V3TZNHrhb-1pnwKZvLCw-Ga56dx1pcb?usp=sharing
    

## Run train scripts

    sh scripts/train_ontal.sh

## Run test scripts

    sh scripts/eval_ontal.sh

## File structure

    ── data/ <br/>
    │   │── thumos_all_feature_val_V3.pickle <br/>
    │   │── thumos_all_feature_test_V3.pickle <br/>
    │   └── thumos14_v2.json <br/>
    ── checkpoint/ <br/>
    │   └── best_epoch.pth <br/>
    ── criterion/ <br/>
    ── Evaluation/ <br/>
    ── models/ <br/>
    ── scripts/ <br/>
    ── util/ <br/>
    dataset.py <br/>
    eval.py <br/>
    main.py <br/>
    on_tal_task.py <br/>
    requirements.txt <br/> 
    readme.md <br/> 
