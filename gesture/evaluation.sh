# Modular

# ground state

python eval_all.py \
	--model='Modular' \
	--feature-maps 64 64 64 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/CoordinationDict4.15M_63.26.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/UnderstandDict95_1.24e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="modular"

# understand network

python eval_all.py \
	--model='Modular' \
	--feature-maps 64 64 64 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/CoordinationDict4.15M_63.26.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/UnderstandDict95_1.24e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="modular" \
	--use-understand

# dynamic tests

python eval_all.py \
		--model='Modular' \
		--feature-maps 64 64 64 \
		--hidden=256 \
		--update-target=10 \
		--MAX_TIME=1200 \
		--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
		--state-dict-path="/home/erik/DATA/Reacher/experiments/All/CoordinationDict4.15M_63.26.pt" \
		--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/UnderstandDict95_1.24e-05.pt" \
		--log-dir="/home/erik/DATA/Reacher/tests/dyn_vid" \
		--render \
		--record \
		--record-name="modular" \
		--use-understand

# Semi modular

python eval_all.py \
	--model='Semimodular' \
	--feature-maps 64 64 32 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/SemimodularDict3.95M_49.812.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/UnderstandDict95_1.24e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Semimodular"

python eval_all.py \
	--model='Semimodular' \
	--feature-maps 64 64 32 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/SemimodularDict3.95M_49.812.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/All/UnderstandDict95_1.24e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Semimodular" \
	--use-understand

python eval_all.py \
	--model='Combine' \
	--feature-maps 64 64 32 \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Reacher/experiments/All/Targets_Reacher_S6_O40-40-3_n1000_0.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/All/CombineDict3.9M_53.956.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests/videos" \
	--render \
	--record \
	--record-name="Combine"



# Humanoid ---------------------------
python eval_all.py \
	--model='Modular' \
	--env-id='Humanoid' \
	--video-h=64 --video-w=64 \
	--feature-maps 64 64 64 \
	--hidden=512 \
	--update-target=300 \
	--MAX_TIME=1200 \
	--test-target-path="/home/erik/DATA/Humanoid/SocialHumanoid_S(18,)_O64-64-3_n100000_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/experiments/Feb16/Humanoid/Coordination/checkpoints/BestDictCombi3350528_98.57.pt" \
	--log-dir="/home/erik/DATA/Humanoid/tests/videos" \
	--render \
	--record \
	--record-name="modular"




	--state-dict-path="/home/erik/DATA/Humanoid/experiments/CoordinationDict2.56M_104.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/BestUnderDict98_0.00010173993566340846.pt" \

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
	--log-dir="/home/erik/DATA/Humanoid/tests/videos" \
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
	--test-target-path="/home/erik/DATA/Humanoid/tests/SocialHumanoid_S(18,)_O64-64-3_n20_0.h5" \
	--state-dict-path="/home/erik/DATA/Humanoid/SemiModularHumanoid2957312_83.833.pt" \
	--state-dict-path2="/home/erik/DATA/Humanoid/BestUnderDict98_0.00010173993566340846.pt" \
	--log-dir="/home/erik/DATA/Humanoid/tests/videos" \
	--render \
	--record \
	--record-name="Combine"


# Pepper --------------------

python eval_pepper.py \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=600 \
	--PORT=41643 \
	--state-dict-path="/home/erik/DATA/Pepper/6.5h_absRew_oneTarget/dict_297984.pt" \
	--log-dir="/home/erik/DATA/Pepper/tests" 


python eval_pepper.py \
	--hidden=256 \
	--update-target=300 \
	--MAX_TIME=600 \
	--PORT=41643 \
	--state-dict-path="/home/erik/DATA/Pepper/11.5h_velRew_oneTarget/dict_534528.pt" \
	--log-dir="/home/erik/DATA/Pepper/tests" 


python eval_pepper.py \
	--hidden=256 \
	--update-target=100 \
	--MAX_TIME=1000 \
	--PORT=41643 \
	--state-dict-path="/home/erik/DATA/Pepper/16h_velRew_randomTarget/dict_759808.pt" \
	--log-dir="/home/erik/DATA/Pepper/tests" \
	--random



