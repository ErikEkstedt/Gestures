# Todo

* Model
	* [ ] Results from trained model + cnn_translation
	* [ ] Using stateless targets. See results. training now
	* [x] Translation in combine training

* [ ] Clean social.py
	* [x] render functions
	* [x] target datatype - consistency. NUMPY IN ENV

* [x] Training loop for social.Social
	* [x] record directly from test


## Training

```bash
python main.py \
--test-thresh=1000000 \       # threshold to not test (save computation)
--num-frames=5000000 \        # total number of frames in training
--num-test=5 \                # how many episodes during each test
--test-interval=30 \          # updates in between tests
--target-path="../results/" \ # path to targets
--record \                    # records all tests
--use-state-target \          # uses the target state as input

```

Run without test or visdom logger and using standard settings
```bash
python main.py --no-vis --no-test
```

## Enjoy

```bash
python enjoy.py \
--render \                      
--record \                      
--MAX_TIME=3000\                        # Number of frames to record/render/run
--update-target=3\                      # Agent gets 3 frames before target is updated
--random-targets \                      # Use if the targets are random and not sequential
--state-dict-path=/path/to/statedict/ \ # state dict to use
--target-path=/path/to/target  \        # Path to target dataset to mimic
```


