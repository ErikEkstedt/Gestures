python main.py --num-frames=3000000 \
	--test-thresh=1500000 \
	--env-id="Reacher" \
	--feature-maps 64 128 256 \
	--hidden=256 --pi-lr=1e-3 \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--num-proc=4 --test-interval=200000 --record


# Understanding

python Understand_main.py --epochs=100 \
	--feature-maps 64 64 64 \
	--hidden=256 --pi-lr=1e-4 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n500000_0.h5" \
	--val-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--test-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Understanding" \
	--batch-size=1024 \
	--num-proc=4 --vis-interval=1 --test-interval=5  --save-interval=5


python Understand_main.py --epochs=100 \
	--feature-maps 64 64 64 \
	--hidden=256 --pi-lr=1e-4 \
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n200000_0.h5" \
	--val-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n100000_1.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Understanding" \
	--batch-size=1024 \
	--num-proc=4 --save-interval=5

python Understand_main.py --epochs=100 \
	--feature-maps 64 64 64 \
	--hidden=256 --pi-lr=1e-4 \
	--train-target-path="/home/erik/com_sci/Master_code/Gestures/gesture/dummy_data/Reacher/Reacher_S6_O40-40-3_n1000_0.h5" \
	--val-target-path="/home/erik/com_sci/Master_code/Gestures/gesture/dummy_data/Reacher/Reacher_S6_O40-40-3_n1000_0.h5" \
	--test-target-path="/home/erik/com_sci/Master_code/Gestures/gesture/dummy_data/Reacher/Reacher_S6_O40-40-3_n1000_0.h5" \
	--log-dir="/home/erik/DATA/Reacher/experiments"\
	--env-id="Understanding" \
	--num-proc=4 --test-interval=5  --save-interval=10

