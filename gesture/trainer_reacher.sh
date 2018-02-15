python main.py --num-frames=5000000 \
	--test-thresh=1500000 \
	--env-id="Reacher" \
	--feature-maps 64 128 256 \
	--hidden=256 --pi-lr=1e-3 \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--num-proc=4 --test-interval=200000 --record


# Understanding
python Understand_main.py --epochs=200 \
	--model='Understand' \
	--feature-maps 64 64 64 \
	--hidden=256 --pi-lr=1e-5 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n200000_0.h5" \
	--val-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Understanding" \
	--batch-size=1024 \
	--num-proc=4 --save-interval=5

# Continue Understand

python Understand_main.py --epochs=200 \
	--model='Understand' \
	--feature-maps 64 64 64 \
	--hidden=256 --pi-lr=1e-5 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n200000_0.h5" \
	--val-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Understanding" \
	--batch-size=1024 \
	--num-proc=4 --save-interval=5 --continue-training \
	--state-dict-path= HEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEERE

# Coordination

python Coordination_main.py --num-frames=5000000 \
	--model='Coordination' \
	--hidden=256 --pi-lr=1e-4 \
	--test-thresh=500000 \
	--test-interval=200000 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n200000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Coordination" \
	--num-proc=4 --save-interval=5 --vis-interval=1


# Semi-Combine

python main.py --num-frames=5000000 \
	--model='SemiCombine' \
	--hidden=256 --pi-lr=1e-4 \
	--test-thresh=500000 \
	--test-interval=200000 \
	--model="SemiCombine"
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="reacher" \
	--num-proc=4 --save-interval=5

# Combine

python main.py --num-frames=5000000 \
	--model='Combine' \
	--hidden=256 --pi-lr=1e-4 \
	--test-thresh=500000 \
	--test-interval=200000 \
	--model="Combine"
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments/Combine"\
	--env-id="Reacher" \
	--num-proc=4 --save-interval=5

