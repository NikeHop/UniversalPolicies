GO_TO_IRRELEVANT_ACTIONS = [Actions(3), Actions(4), Actions(5), Actions(6)]
BOSSLEVEL_IRRELEVANT_ACTIONS = [Actions(6)]

def check_subsequence(subsequence, action_space):
    if len(subsequence) < 3:
        return False
    print([int(elem) for elem in subsequence])
    if action_space == 0:
        # It should contain either a turn left or turn right
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 1 in action_types or 0 in action_types:
            return True

    elif action_space == 1:
        # It should contain three right turns
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 1 in action_types and len(action_types) == 1:
            return True

    elif action_space == 2:
        # It should contain three left turns
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        if 0 in action_types and len(action_types) == 1:
            return True

    elif action_space == 3:
        # It should contain a diagonal
        subsequence = [int(elem) for elem in subsequence]
        action_types = set(subsequence)
        print(action_types)
        if 7 in action_types or 8 in action_types:
            return True

    return False


def check_condition(examples, n_examples):
    print("Check conditions")
    if len(examples) < 4:
        return False

    for i, value in examples.items():
        print(i, len(value))
        if len(value) < n_examples:
            return False
    return True


def generate_example_trajectories(config):
    if config["trajectory_type"]=="full_action_space_single_random":
        generate_full_action_space_random(
            config["env"],
            config["seed"],
            config["use_img_obs"],
            config["num_distractors"],
            config["img_size"],
            config["n_action_spaces"],
            config["save_directory"],
            config["n_examples"]
        )
    elif config["trajectory_type"] == "every_action":
        generate_examples(config)
    else:
        raise NotImplementedError(f"This trajectory type has not been implemented {config['trajectory_type']}")

def generate_full_action_space_random(
    env_name: str,
    seed: int,
    use_img_obs: bool,
    num_distractors: int,
    img_size: int,
    n_action_spaces: int,
    save_directory: str,
    n_examples: int
):
    """
    Generate a single episode of demonstrations from the BabyAIBot

    Args:
        env_name (str): the name of the environment (without prefixes and suffixes)
        action_space (ActionSpace): action space
        minimum_length (int): minimum length of the episode (otherwise it is discarded)
        seed (int): seed
        use_img_obs (bool): if true uses image observation otherwise uses fully obs state space
        debug (bool): debug flag
        img_size (int): size of the image, only relevant if use_img_obs is true
    """
    # Seed everything
    set_seed(seed)

    # Set up datastructure 
    example_contexts = defaultdict(list)

    # Iterate over the action spaces
    for _ in range(n_examples):
        for action_space in range(n_action_spaces):
            action_space = ActionSpace(action_space)
            legal_actions = action_space.get_legal_actions()

            if "GOTO"==env_name:
                # Sample an environment 
                env_id = random.choice(FIXINSTGOTO_ENVS)
                env = gym.make(
                    env_id,
                    highlight=False,
                    action_space=action_space,
                    num_dists=num_distractors,
                )
                irrelevant_actions = GO_TO_IRRELEVANT_ACTIONS
            elif "BossLevel" in env_name:
                env = gym.make(
                    env_name,
                    highlight=False,
                    action_space=action_space,
                )
                irrelevant_actions = BOSSLEVEL_IRRELEVANT_ACTIONS
            else:
                raise NotImplementedError(
                    f"This environment has not been implemented {env_name}"
                )

            # Apply Wrapper (image or state as obs)
            if use_img_obs:
                env = RGBImgObsWrapper(env, tile_size=img_size // env.unwrapped.width)
            else:
                env = FullyObsWrapper(env)

            success = False 
            while not success:
                obs = env.reset()[0]
                images = [obs["image"]]
                executed_actions = []
                while len(executed_actions)<len(legal_actions):
                    # Get possible actions at current state

                    possible_actions = get_possible_actions(obs["image"])
                    legal_possible_actions = [action for action in legal_actions if action in possible_actions and action not in executed_actions]

                    if len(legal_possible_actions)==0:
                        break

                    for action in legal_possible_actions:
                        if action in irrelevant_actions:
                            executed_actions.append(action)
                            continue

                        new_obs, reward, done, _, _ = env.step(action)
                        obs = new_obs
                        images.append(obs["image"])
                        executed_actions.append(action)

                
                if len(executed_actions)==len(legal_actions):
                    success = True
                    compressed_video = blosc.pack_array(np.stack(images,axis=0))
                    example_contexts[action_space].append(compressed_video)
                else:
                    seed += 1

    # Save example contexts
    filename = f"{env_name}_{n_examples}_{num_distractors}_full_action_space_random.pkl"
    with open(os.path.join(save_directory,filename),"wb") as file:
        pickle.dump(example_contexts,file)
    
    # Get longest episode
    for action_space, examples in example_contexts.items():
        longest_episode = max([blosc.unpack_array(example).shape[0] for example in examples])
        print(f"Action space {action_space} longest episode {longest_episode}")

def generate_examples(config):
    # Prepare data structure
    example_contexts = defaultdict(list)
    save_directory = config["save_directory"]

    # Generate a start state
    env_id = FIXINSTGOTO_ENVS[0]
    env = gym.make(
        env_id,
        highlight=False,
        action_space=ActionSpace.all,
        num_dists=0,
        action_space_agent_color=True,
    )

    env = FullyObsWrapper(env)
    _ = env.reset()[0]
    grid = env.grid
    cleaned_grid = []
    for elem in grid.grid:
        if isinstance(elem,minigrid.core.world_object.Wall):
            cleaned_grid.append(elem)
        elif elem is None:
            cleaned_grid.append(elem)
        else:
            cleaned_grid.append(None)

    grid.grid = cleaned_grid
    env.set_state(grid,(3,3),0,0)
    
    # All actions to test
    actions = list(range(4)) + list(range(7,19))
    for action_space in ActionSpace:
        legal_actions = action_space.get_legal_actions()
        grid.grid = cleaned_grid
        env.set_state(grid,(3,3),3,0)
        obs = env.step(6)
        state = obs[0]["image"]
        example_contexts[action_space].append(state)

        if action_space == ActionSpace.all:
            continue
        
        for action in actions:
            grid.grid = cleaned_grid
            env.set_state(grid,(3,3),3,0)
            if action not in legal_actions:
                action = 6
            obs = env.step(action)
            state = obs[0]["image"]
            example_contexts[action_space].append(state)
    
    # Format the contexts
    for action_space, examples in example_contexts.items():
        example_contexts[action_space] = [blosc.pack_array(np.stack(examples,axis=0))]

    # Save examples
    filename = f"no_padding_each_action_0.pkl"
    with open(os.path.join(save_directory, filename), "wb") as file:
        pickle.dump(example_contexts, file)


def get_possible_actions(obs):
    """
    Get the possible actions at the current state
    """
    possible_actions = []
    # extract agent position and direction
    (x, y), direction = extract_agent_pos_and_direction(obs)
    for action in Actions:
        if action == 0:
            # Left, always possible
            possible_actions.append(action)
        elif action == 1:
            # Right, always possible
            possible_actions.append(action)
        elif action == 2:
            # Forward, check if there is a wall in front
            if direction == 0:
                # Facing west
                if x <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2:
                    possible_actions.append(action)
        elif action == 3:
            # Pickup
            possible_actions.append(action)
        elif action == 4:
            # Drop
            possible_actions.append(action)
        elif action == 5:
            # Toggle
            possible_actions.append(action)
        elif action == 6:
            # Done
            possible_actions.append(action)
        elif action == 7:
            # Diagonal left
            if direction == 0:
                # Facing west
                if x <= 5 and y >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5 and x <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2 and y <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2 and x >= 2:
                    possible_actions.append(action)
        elif action == 8:
            # Diagonal right
            if direction == 0:
                # Facing west
                if x <= 5 and y <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y <= 5 and x >= 2:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x >= 2 and y >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y >= 2 and x <= 5:
                    possible_actions.append(action)
        elif action == 9 or action == 17:
            # Right move
            if x <= 5:
                possible_actions.append(action)
        elif action == 10:
            # Down move
            if y <= 5:
                possible_actions.append(action)
        elif action == 11 or action == 16:
            # Left move
            if x >= 2:
                possible_actions.append(action)
        elif action == 12:
            # Up move
            if y >= 2:
                possible_actions.append(action)
        elif action == 13:
            # Turn around, always possible
            possible_actions.append(action)
        elif action == 14:
            # Diagonal backwards left, same as diagonal left if we would turn 180 degrees
            if direction == 0:
                # Facing west
                if x >= 2 and y >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2 and x <= 5:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x <= 5 and y <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5 and x >= 2:
                    possible_actions.append(action)
        elif action == 15:
            # Diagonal backwards right, same as diagonal right if we would turn 180 degrees
            if direction == 0:
                # Facing west
                if x >= 2 and y <= 5:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2 and x >= 2:
                    possible_actions.append(action)
            elif direction == 2:
                # Facing east
                if x <= 5 and y >= 2:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5 and x <= 5:
                    possible_actions.append(action)
        elif action == 18:
            # Backward, same as forward if we would turn 180 degrees
            if direction == 2:
                # Facing east
                if x <= 5:
                    possible_actions.append(action)
            elif direction == 3:
                # Facing north
                if y <= 5:
                    possible_actions.append(action)
            elif direction == 0:
                # Facing west
                if x >= 2:
                    possible_actions.append(action)
            elif direction == 1:
                # Facing south
                if y >= 2:
                    possible_actions.append(action)

    return possible_actions