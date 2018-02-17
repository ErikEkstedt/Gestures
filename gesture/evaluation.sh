# Modular

python eval_all.py \
	--model='Modular' \
	--feature-maps 64 64 64 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/Coordination/checkpoints/BestDictCombi4153344_63.26.pt"\
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="modular"

python eval_all.py \
	--model='Semimodular' \
	--feature-maps 64 64 32 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/SemiCombine5M/checkpoints/BestDictCombi3948544_49.812.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Semimodular"


python eval_all.py \
	--model='Combine' \
	--feature-maps 64 64 32 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/Combine/checkpoints/BestDictCombi3948544_53.956.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/Understanding/Dict1.243e-05/run-3/checkpoints/BestUnderDict95_1.2435867166478063e-05.pt"\
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="modular"

# Humanoid

python eval_all.py \
	--model='Modular' \
	--env-id='Humanoid' \
	--feature-maps 64 64 64 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/BestDictCombi3350528_98.57.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/BestUnderDict98_0.00010173993566340846.pt" \
	--log-dir="/home/erik/DATA/Reacher/tespts/videos" \
	--render \
	--record \
	--record-name="modular"

python eval_all.py \
	--model='Semimodular' \
	--env-id='Humanoid' \
	--feature-maps 128 128 128 \
	--video-h=64 --video-w=64 \
	--hidden=512 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/SemiModularHumanoid2957312_83.833.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/BestUnderDict98_0.00010173993566340846.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Semimodular"

python eval_all.py \
	--model='Combine' \
	--env-id='Humanoid' \
	--feature-maps 128 128 128 \
	--video-h=64 --video-w=64 \
	--hidden=512 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/SemiModularHumanoid2957312_83.833.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/BestUnderDict98_0.00010173993566340846.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Combine"

