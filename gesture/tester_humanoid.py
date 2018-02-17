# Modular

python eval_modular.py \
	--feature-maps 128 128 128 --render \
    --model="Modular" \
	--hidden=256 --update-target=300 --MAX_TIME=10000 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n50000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/experiments/Feb16/Humanoid/Coordination/checkpoints/BestDictCombi3350528_98.57.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/experiments/Feb16/Humanoid/understand/run-0/checkpoints/BestUnderDict98_0.00010173993566340846.pt"\
	--log-dir="/home/erik/DATA/Humanoid/tests" \
	--env-id="Evaluate_Modular" \
	--record --njoints=6

# SemiCombine

python eval_semicombine.py \
	--feature-maps 64 64 32 --render \
	--hidden=256 --update-target=300 --MAX_TIME=10000\
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb14/SemiCombine5M/checkpoints/BestDictCombi3948544_49.812.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/Feb14/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_semicombine" \
	--record


# Combine

python eval_combine.py \
	--feature-maps 64 64 32 --render \
	--hidden=256 --update-target=300 --MAX_TIME=10000\
	--train-target-path="/home/erik/DATA/Reacher/experiments/Feb14/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb14/Combine/checkpoints/BestDictCombi3948544_53.956.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_semicombine" \
	--record

# HUMANOID

python eval_modular.py \
	--feature-maps 64 64 32 --render \
    --model="Modular" \
	--hidden=256 --update-target=300 --MAX_TIME=10000 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n50000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/experiments/Feb16/Humanoid/Coordination/checkpoints/BestDictCombi3350528_98.57.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/experiments/Feb16/Humanoid/understand/run-0/checkpoints/BestUnderDict98_0.00010173993566340846.pt" \
	--log-dir="/home/erik/DATA/Humanoid/tests" \
	--env-id="Evaluate_Modular" \
	--record

# SemiCombine

python eval_semicombine.py \
	--feature-maps 64 64 32 --render \
	--hidden=256 --update-target=300 --MAX_TIME=10000\
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb14/SemiCombine5M/checkpoints/BestDictCombi3948544_49.812.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/Feb14/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_semicombine" \
	--record


# Combine

python eval_combine.py \
	--feature-maps 64 64 32 --render \
	--hidden=256 --update-target=300 --MAX_TIME=10000\
	--train-target-path="/home/erik/DATA/Reacher/experiments/Feb14/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb14/Combine/checkpoints/BestDictCombi3948544_53.956.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_semicombine" \
	--record
