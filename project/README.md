## Collect Random Target Poses

First cd into /project/project/data, then:
```bash
python collect_targets.py \
--env-id=Humanoid \                # environment to use
--dpoints=500000 \                 # how many targets to collect
--filepath=/PATH/TO/filedirectoy \ # Filepath to file directory
--video-w=40 \                     # video width
--video-h=40 \                     # video height
```

## Training

Start a training
```bash
python main.py \
--test-thresh=1000000 \                 # threshold to not test (save computation)
--num-frames=5000000 \                  # total number of frames in training
--num-test=5 \                          # how many episodes during each test
--test-interval=30 \                    # updates in between tests
--train-target-path="/PATH/TO/Targets/" # path to targets
--test-target-path="/PATH/TO/Targets/"  # path to targets
--record \                              # records all tests
--use-state-target \                    # uses the target state as input
```

Continue from a checkpoint
```bash
python main.py \
--continue-training                     # Loads a state dict and continues training
--test-thresh=1000000 \                 # threshold to not test (save computation)
--num-frames=5000000 \                  # total number of frames in training
--num-test=5 \                          # how many episodes during each test
--test-interval=30 \                    # updates in between tests
--train-target-path="/PATH/TO/Targets/" # path to targets
--test-target-path="/PATH/TO/Targets/"  # path to targets
--record \                              # records all tests
--use-state-target \                    # uses the target state as input
```

Run without test or visdom logger and using standard settings 
envronment : SocialReacher
targets    : small set in repo
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
--state-dict-path=/path/to/statedict/ \ # state dict to use
--target-path=/path/to/target  \        # Path to target dataset to mimic
```


