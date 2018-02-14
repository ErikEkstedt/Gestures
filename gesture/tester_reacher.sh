# Modular

python eval_modular.py \
	--feature-maps 64 64 64 --render \
	--hidden=256 --update-target=300 --MAX_TIME=10000\
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb13/Modular/BestDictCombi2383872_50.632.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/Feb13/Modular/BestUnderDict95_1.2435867166478063e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_Modular" 

# SemiCombine

python eval_semicombine.py \
	--feature-maps 64 64 64 --render \
	--hidden=256 --update-target=300 --MAX_TIME=1000\
	--train-target-path="/home/erik/DATA/Reacher/SocialReacher_S(6,)_O40-40-3_n50000_1.h5" \
	--state-dict-path="/home/erik/DATA/Reacher/experiments/Feb13/semiModular/SemiCombine/run-0/checkpoints/dict_2965504_TEST_42.391.pt" \
	--state-dict-path2="/home/erik/DATA/Reacher/experiments/Feb13/Modular/BestUnderDict95_1.2435867166478063e-05.pt" \
	--log-dir="/home/erik/DATA/Reacher/tests" \
	--env-id="Evaluate_semicombine" 



