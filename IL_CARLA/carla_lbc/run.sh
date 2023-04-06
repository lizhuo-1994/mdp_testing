# python benchmark_agent.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 12
python test_gen.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 2 --method generative 
python test_gen.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 2 --method generative+density 
python test_gen.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 2 --method generative+sensitivity
python test_gen.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 2 --method generative+performance
python test_gen.py --suite=town2 --model-path=ckpts/image/model-10.th --emguide --hour 2 --method generative+novelty 