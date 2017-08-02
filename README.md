# C51DQN
#### A TensorFlow implementation of DeepMind's "A Distributional Perspective on Reinforcement Learning".(C51-DQN)

##### I use the atari game Breakout-v0 for the test.

##### The feedback after an action contains 4 parts:
    state, real-reward, game_over, lives_rest

##### There are 4 actions in the game Breakout-v0:
    0: hold and do nothing
    1: throw the ball
    2: move right
    3: move left
    
##### Note: The train-reward is equal to the real-reward in most of the time. But in the beginning of the new live, Its obvious the action should be throwing the ball. So the train-reward should be 1 in spite of the real-reward of throwing is zero. And the train-reward is -1 when you miss the ball.

##### Train the net only when we score or miss the ball.
    if train_reward != 0:  # train when miss the ball or score or throw the ball in the beginning
        print 'len(items_buffer):', len(items_buffer)
        for item in items_buffer:
            agent.train(item[0], -1 if throw else train_reward, item[1], item[2], item[3])
        items_buffer = []            