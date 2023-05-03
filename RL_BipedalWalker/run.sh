# python enjoy.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --em --no-render --hour 12
# python test_gen.py --method generative+novelty --hour 12
# python test_gen.py --method generative --hour 12
# python test_gen.py --method generative+density --hour 12
# python test_gen.py --method generative+sensitivity --hour 12
# python test_gen.py --method generative+performance --hour 12

#python mdpfuzz.py --alg tqc --env BipedalWalkerHardcore-v3 --folder rl-trained-agents/ --em --no-render --hour 1

python test_gen.py --method generative+novelty --hour 2 --step 50
python test_gen.py --method generative+novelty --hour 2 --step 100
python test_gen.py --method generative+novelty --hour 2 --step 200