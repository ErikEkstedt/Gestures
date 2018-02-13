python main.py --num-frames=10000000 \
	--test-thresh=2000000 \
	--feature-maps 64 128 256 \
	--hidden=512 --pi-lr=1e-3 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n100000_0.ht" \
	--train-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n200000_0.ht" \
	--log-dir="/home/erik/"\
	--num-proc=12 --test-interval=400000 --record

python main.py --num-frames=10000000 \
	--test-thresh=2000000 \
	--feature-maps 64 128 256 \
	--hidden=512 --pi-lr=5e-4 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n100000_0.ht" \
	--train-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n200000_0.ht" \
	--log-dir="/home/erik/"\
	--num-proc=12 --test-interval=400000 --record


python main.py --num-frames=10000000 \
	--test-thresh=2000000 \
	--feature-maps 64 128 256 \
	--hidden=512 --pi-lr=1e-4 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n100000_0.ht" \
	--train-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n200000_0.ht" \
	--log-dir="/home/erik/"\
	--num-proc=12 --test-interval=400000 --record

python main.py --num-frames=10000000 \
	--test-thresh=2000000 \
	--feature-maps 64 128 256 \
	--hidden=512 --pi-lr=5e-5 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n100000_0.ht" \
	--train-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O80-80-3_n200000_0.ht" \
	--log-dir="/home/erik/"\
	--num-proc=12 --test-interval=400000 --record
