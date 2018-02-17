python train_semicombine.py --num-frames=5000000 \
	--num-proc=4 \
	--env-id='Humanoid' \
	--model='SemiCombine' \
	--feature-maps 128 128 128 \
	--hidden=512 \
	--pi-lr=1e-4 \
	--test-thresh=500000 \
	--test-interval=200000 \
	--train-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n200000_0.h5"\
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n100000_0.h5" \
	--log-dir="/home/erik/DATA/Humanoid/experiments"

python main.py --num-frames=5000000 \
	--model='SemiCombine' \
	--feature-maps 128 128 128 \
	--hidden=512 \
	--pi-lr=1e-4 \
	--test-thresh=500000 \
	--test-interval=200000 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="reacher" \
	--num-proc=4 
