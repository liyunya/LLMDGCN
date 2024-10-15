## LLMDGCN

Source code for CIKM2024 "LLM-Empowered Few-Shot Node Classification on Incomplete Graphs with Real Node Degrees"

## Environment Settings
> python==3.8.10 \
> torch==2.0.0 \
> numpy==1.24.2 \
> torch-cluster==1.6.3 \
> torch-geometric==2.6.1 \
> torch-scatter==2.1.2 \
> torch-sparse==0.6.18 \
> torch-spline-conv==1.2.2 \
> torchmetrics==1.4.3 \
> scipy==1.10.1


## Usage

You can use the following commend to run the code; 

> python edge.py --dataset citeseer --few_shot_perclass 0 --thred_conf 0.9 --thred_sim 0.7 --topk 10

