from pydantic import Field
import random
import numpy as np
from PIL import Image as PILImage
import gym
import gym_sokoban
import io
import os
import json
import argparse
from typing import List, Literal, Tuple, Union
from mcp.types import TextContent
from fastmcp import FastMCP
from fastmcp.utilities.types import Image
from fastmcp.server.dependencies import get_http_headers
from fastmcp.tools.tool import ToolResult
import fastmcp

from deadlock_detector.interface import check_deadlock

def apply_sokoban_monkey_patch():
    """
    Apply monkey patch to SokobanEnv to store and retrieve difficulty score.
    """
    import gym
    import gym_sokoban
    from gym_sokoban.envs import sokoban_env, room_utils

    # Save the original reset method
    _original_reset = sokoban_env.SokobanEnv.reset

    def patched_reset(self, second_player=False, render_mode='rgb_array'):
        """
        Patched reset method that stores the difficulty score after reset.
        """
        result = _original_reset(self, second_player, render_mode)
        # Store the difficulty score from global variable
        self.difficulty_score = getattr(room_utils, 'best_room_score', 0)
        return result

    def get_difficulty_score(self):
        """
        Retrieve the current level's difficulty score.
        """
        return getattr(self, 'difficulty_score', 0)

    # Apply monkey patch
    sokoban_env.SokobanEnv.reset = patched_reset
    sokoban_env.SokobanEnv.get_difficulty_score = get_difficulty_score

    print("✅ Monkey patch applied!")

apply_sokoban_monkey_patch()

fastmcp.settings.log_level = 'CRITICAL'
# np.bool8 = np.bool  # fix for old gym

# ------- Global Variables -------
ARGS = None

# ------- Global Game State Management -------
class SokobanGame:
    def __init__(self):
        self.initialized = False
        self.env = None
        self.step_count = 0
        self.accumulated_reward = 0
        self.done = False
        self.img_up_scale = 1.75
        self.move_history = []
        self.seed = None
        self.room_id = None
        self.init_state_reward = 3.0
        self.first_move_reward = 3.0

    def observation_to_image(self, observation: np.ndarray) -> Image:
        """Convert observation to FastMCP Image"""
        if observation is None or not isinstance(observation, np.ndarray):
            raise ValueError("Observation must be a valid numpy array")
            
        if observation.dtype != np.uint8:
            observation = observation.astype(np.uint8)
                
        pil_image = PILImage.fromarray(observation)
        new_size = (int(pil_image.width * self.img_up_scale), int(pil_image.height * self.img_up_scale))
        pil_image = pil_image.resize(new_size, PILImage.Resampling.NEAREST)
                
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_bytes = buffer.getvalue()
                
        return Image(data=img_bytes, format="jpeg")

    def initialize(self, seed, room_id='Sokoban-small-v0', stage_width=5, stage_height=5, num_boxes=1, num_gen_steps=None, min_difficulty_score=None):
        try:
            self.env = gym.make(room_id)
            # safe attribute setting
            if hasattr(self.env, 'env') and hasattr(self.env.env, 'env'):
                env_instance = self.env.env.env
                if hasattr(env_instance, 'dim_room'):
                    env_instance.dim_room = (stage_width, stage_height)
                if hasattr(env_instance, 'num_boxes'):
                    env_instance.num_boxes = num_boxes
                if hasattr(env_instance, 'num_gen_steps'):
                    if num_gen_steps is not None:
                        env_instance.num_gen_steps = num_gen_steps
                    else:
                        env_instance.num_gen_steps = int(1.7 * (stage_width + stage_height))
            
            random.seed(seed)
            np.random.seed(seed)
            self.env.reset()
            if min_difficulty_score is not None:
                while self.env.get_difficulty_score() < min_difficulty_score:
                    self.env.reset()
            self.seed = seed
            self.room_id = room_id
            self.initialized = True
            self.accumulated_reward = self.init_state_reward
        except Exception as e:
            raise RuntimeError(f"Failed to initialize environment: {e}")

    def get_state_image(self) -> Image:
        rendered_obs = self.env.render(mode='rgb_array')
        return self.observation_to_image(rendered_obs)

    def get_state_str(self) -> str:
        digits_to_chars = {0: 'X', 1: '.', 2: 'T', 3: '*', 4: 'B', 5: '@', 6: '+'}
        
        try:
            room_state = self.env.env.env.room_state.copy()
            player_x, player_y = self.env.env.env.player_position
            if self.env.env.env.room_fixed[player_x, player_y] == 2:
                room_state[player_x, player_y] = 6
            state_str = ''
            for l in room_state:
                state_str += ' '.join([digits_to_chars[c] for c in l])
                state_str += '\n'
            return state_str
        except (AttributeError, KeyError) as e:
            return f"Error getting game state: {e}"

    def step(self, direction: Literal["up", "down", "left", "right"], render_mode: Literal["image", "text"] = "image") -> Tuple[Union[Image, str], float, bool, dict]:
        action = {"up": 1, "down": 2, "left": 3, "right": 4}[direction.lower()]
        observation, reward, done, info = self.env.step(action)
        
        if self.step_count == 0:
            reward += self.first_move_reward
        
        self.step_count += 1
        self.accumulated_reward += reward
        self.done = done
        self.move_history.append(action)
        
        if render_mode == "image":
            return self.observation_to_image(observation), reward, done, info
        elif render_mode == "text":
            return self.get_state_str(), reward, done, info

sokoban_game = SokobanGame()
mcp = FastMCP(name="Sokoban Game Server")

def safe_get_arg(attr_name, env_name, default_value, convert_func=None):
    """Get argument safely"""
    if ARGS and hasattr(ARGS, attr_name):
        value = getattr(ARGS, attr_name)
        if value is not None:
            return value
    
    if env_name in os.environ:
        try:
            value = os.environ[env_name]
            return convert_func(value) if convert_func else value
        except (ValueError, TypeError):
            pass  # Use default value
    
    return default_value

@mcp.prompt
def get_init_state_prompt() -> str:
    """Generate sokoban game and ask model to solve it"""
    seed = safe_get_arg('seed', 'ENV_SEED', None, int)
    room_id = safe_get_arg('room_id', 'ENV_ROOM_ID', 'Sokoban-small-v0')
    stage_width = safe_get_arg('stage_width', 'ENV_STAGE_WIDTH', 7, int)
    stage_height = safe_get_arg('stage_height', 'ENV_STAGE_HEIGHT', 7, int)
    num_boxes = safe_get_arg('num_boxes', 'ENV_NUM_BOXES', 2, int)
    num_gen_steps = safe_get_arg('num_gen_steps', 'ENV_NUM_GEN_STEPS', -1, int)
    num_gen_steps = None if num_gen_steps == -1 else num_gen_steps
    min_difficulty_score = safe_get_arg('min_difficulty_score', 'ENV_MIN_DIFFICULTY_SCORE', None, int)

    if seed is None:
        return "Error: Seed must be provided either as --seed argument or ENV_SEED environment variable."
        
    try:
        sokoban_game.initialize(seed, room_id, stage_width, stage_height, num_boxes, num_gen_steps, min_difficulty_score)
    except Exception as e:
        return f"Error initializing game: {e}"

    # prompt = (
    #     "You are about to play a Sokoban puzzle game. Your objective is to solve the puzzle "
    #     "by moving all the boxes onto their designated target locations. To make moves, use "
    #     "the 'move_sequence' command and provide your chosen sequence of moves.\n\n"
    #     "Below is the initial state of the game:\n"
    #     f"{sokoban_game.get_state_str()}\n"
    #     "\n"
    #     "Legend:\n"
    #     "  . : empty floor\n"
    #     "  X : wall\n"
    #     "  B : box\n"
    #     "  @ : player\n"
    #     "  T : target\n"
    #     "  * : box on target\n"
    #     "  + : player on target\n"
    #     "\n"
    #     "Don't overthink it—be proactive and give it a try! "
    #     "Try to solve the puzzle actively, but avoid making moves that would cause a deadlock."
    # )

    # Sokoban Puzzle Game

    prompt = (
        "You are playing a Sokoban puzzle. **Goal**: Push all boxes (B) onto target locations (T).\n\n"
        "## Current Game State:\n"
        "```\n"
        f"{sokoban_game.get_state_str()}\n"
        "```\n\n"
        "## Legend:\n"
        "- `.` : empty floor\n"
        "- `X` : wall  \n"
        "- `B` : box\n"
        "- `@` : player\n"
        "- `T` : target\n"
        "- `*` : box on target\n"
        "- `+` : player on target\n\n"
        "## Game Rules:\n"
        "1. Player can move in 4 directions (up/down/left/right)\n"
        "2. Player can push boxes, but cannot pull them\n"
        "3. **Important**: Boxes on targets can still be pushed\n"
        "4. **Important**: Player can stand on target locations\n"
        "5. Only one box can be pushed at a time\n"
        "6. Avoid deadlocks (boxes stuck in corners/against walls where they can't reach targets)\n\n"
        "## Strategy Guidelines:\n"
        "**Before making moves, think systematically:**\n\n"
        "1. **Analyze the puzzle layout**: Identify the current positions of both boxes and both targets\n"
        "2. **Plan box-target assignments**: Decide which box should go to which target (consider proximity and path obstacles)\n"
        "3. **Visualize the solution path**: Think about the sequence needed to get each box to its target\n"
        "4. **Check for potential deadlocks**: Ensure your planned moves won't create unsolvable situations\n\n"
        "## Current Puzzle Details:\n"
        "- **Grid size**: 7×7\n"
        "- **Boxes**: 2 \n"
        "- **Targets**: 2\n\n"
        "**Start by analyzing which box should go to which target, then plan your first sequence of moves.**\n\n"
        "**Don't overthink it—be proactive and start moving!**"
    )

    return prompt


@mcp.tool
def get_init_state() -> ToolResult | str:
    """Initialize the Sokoban game. Can only be called once."""
    if sokoban_game.initialized:
        return "Error: Game already initialized. Cannot initialize multiple times."
        
    seed = safe_get_arg('seed', 'ENV_SEED', None, int)
    room_id = safe_get_arg('room_id', 'ENV_ROOM_ID', 'Sokoban-small-v0')
    stage_width = safe_get_arg('stage_width', 'ENV_STAGE_WIDTH', 7, int)
    stage_height = safe_get_arg('stage_height', 'ENV_STAGE_HEIGHT', 7, int)
    num_boxes = safe_get_arg('num_boxes', 'ENV_NUM_BOXES', 2, int)
    num_gen_steps = safe_get_arg('num_gen_steps', 'ENV_NUM_GEN_STEPS', -1, int)
    num_gen_steps = None if num_gen_steps == -1 else num_gen_steps
    min_difficulty_score = safe_get_arg('min_difficulty_score', 'ENV_MIN_DIFFICULTY_SCORE', None, int)
        
    if seed is None:
        return "Error: Seed must be provided either as --seed argument or ENV_SEED environment variable."
        
    try:
        sokoban_game.initialize(seed, room_id, stage_width, stage_height, num_boxes, num_gen_steps, min_difficulty_score)
    except Exception as e:
        return f"Error initializing game: {e}"
        
    description = (
        "Sokoban game initialized. Ready to play! "
        "Use move_sequence to make moves.\n\n"
        "Here is the initial state of the game:\n"
        f"{sokoban_game.get_state_str()}\n"
        "\n"
        "Legend:\n"
        "  . : empty floor\n"
        "  X : wall\n"
        "  B : box\n"
        "  @ : player\n"
        "  T : target\n"
        "  * : box on target\n"
    )
    
    metadata = {
        "room_id": sokoban_game.room_id,
        "seed": sokoban_game.seed,
        "step_count": sokoban_game.step_count,
        "accumulated_reward": sokoban_game.accumulated_reward,
        "early_exit": False,
        "move_history": sokoban_game.move_history,
        "is_passed": False,
    }
    
    return ToolResult(
        content=[TextContent(type="text", text=description)],
        structured_content=metadata
    )

@mcp.tool
def move_sequence(
    directions: List[str] = Field(..., description='List of movement directions, e.g. ["up", "down", "left", "right"] (max 5 moves)'),
) -> ToolResult | str:
    """Execute a sequence of moves in the Sokoban game."""
    if not sokoban_game.initialized:
        return "Error: Game not initialized. Call get_init_state first."
        
    if len(directions) > 5:
        return f"Error: Maximum 5 moves allowed, got {len(directions)} moves."
        
    if sokoban_game.done:
        return "Error: Game already completed. Cannot make more moves."
        
    valid_directions = {"up", "down", "left", "right"}
    results = []
    is_passed = False
    is_deadlock = False
    reasons = []
    
    for direction in directions:
        if direction.lower() not in valid_directions:
            results.append(f"Error: Invalid direction '{direction}'. Valid directions: {valid_directions}")
            break
            
        try:
            state_str, reward, done, info = sokoban_game.step(direction, render_mode="text")
            if info['action.moved_box']:
                is_deadlock, reasons = check_deadlock(state_str)
        except Exception as e:
            results.append(f"Error during move: {e}")
            break

        move_text = f"Step {sokoban_game.step_count}, Move: {direction}\n{state_str}\n\n"
        results.append(move_text)

        if is_deadlock:
            # Append deadlock message with formatted reasons (one per line, indented)
            deadlock_msg = "🚫 Deadlock detected!"
            if reasons:
                deadlock_msg += "\n" + "\n".join(f"  • {reason}" for reason in reasons)
            results.append(deadlock_msg)
            is_deadlock = True
            break
                
        if done and reward > 0:
            results.append("🎉 Congratulations! You solved the puzzle! 🎉")
            is_passed = True
            break
        elif done:
            results.append("Game completed.")
            break

    # ------- Add message if not passed or deadlock -------
    if not is_passed and not is_deadlock:
        results.append(
            "Not yet solved, please continue."
        )
    
    metadata = {
        "room_id": sokoban_game.room_id,
        "seed": sokoban_game.seed,
        "step_count": sokoban_game.step_count,
        "accumulated_reward": sokoban_game.accumulated_reward,
        "early_exit": sokoban_game.done or is_deadlock,
        "move_history": sokoban_game.move_history,
        "is_passed": is_passed,
        "is_deadlock": is_deadlock,
    }
        
    return ToolResult(
        content=[TextContent(type="text", text=r) for r in results],
        structured_content=metadata
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sokoban Game MCP Server')
    parser.add_argument('--seed', type=int, help='Random seed for game initialization')
    parser.add_argument('--room-id', dest='room_id', type=str, help='Room ID for Sokoban environment')
    parser.add_argument('--stage-width', dest='stage_width', type=int, help='Width of the game stage')
    parser.add_argument('--stage-height', dest='stage_height', type=int, help='Height of the game stage')
    parser.add_argument('--num-boxes', dest='num_boxes', type=int, help='Number of boxes in the game')
    parser.add_argument('--num-gen-steps', dest='num_gen_steps', type=int, help='Number of generation steps')
    parser.add_argument('--min-difficulty-score', dest='min_difficulty_score', type=int, help='Minimum difficulty score')
        
    ARGS = parser.parse_args()
    mcp.run(transport="stdio")