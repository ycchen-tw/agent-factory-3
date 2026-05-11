from pydantic import Field
import random
import numpy as np
from PIL import Image as PILImage
import gym
import gym_sokoban
import io
import os
import json
from typing import List, Literal, Tuple, Union
from fastmcp import FastMCP
from fastmcp.utilities.types import Image, TextContent
from fastmcp.server.dependencies import get_http_headers

import fastmcp
fastmcp.settings.log_level = 'CRITICAL'

# Global game state management
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
        if observation.dtype != np.uint8:
            observation = observation.astype(np.uint8)
        
        # Convert to PIL Image and resize for better visibility
        pil_image = PILImage.fromarray(observation)
        new_size = (int(pil_image.width * self.img_up_scale), int(pil_image.height * self.img_up_scale))
        pil_image = pil_image.resize(new_size, PILImage.NEAREST)
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        img_bytes = buffer.getvalue()
        
        return Image(data=img_bytes, format="jpeg")
    
    def initialize(self, seed, room_id='Sokoban-small-v0', stage_width=5, stage_height=5, num_boxes=1, num_gen_steps=None):
        self.env = gym.make(room_id)
        self.env.env.env.dim_room = (stage_width, stage_height)
        self.env.env.env.num_boxes = num_boxes
        if num_gen_steps is not None:
            self.env.env.env.num_gen_steps = num_gen_steps
        else:
            self.env.env.env.num_gen_steps = int(1.7 * (stage_width + stage_height))
        random.seed(seed)
        np.random.seed(seed)
        self.env.reset()

        self.seed = seed
        self.room_id = room_id
        self.initialized = True
        self.accumulated_reward = self.init_state_reward

    def get_state_image(self) -> Image:
        rendered_obs = self.env.render(mode='rgb_array')
        return self.observation_to_image(rendered_obs)
    
    def step(self, direction: Literal["up", "down", "left", "right"]) -> Tuple[Image, float, bool, dict]:
        action = {"up": 1, "down": 2, "left": 3, "right": 4}[direction.lower()]
        observation, reward, done, info = self.env.step(action)
        if self.step_count == 0:
            reward += self.first_move_reward

        self.step_count += 1
        self.accumulated_reward += reward
        self.done = done
        self.move_history.append(action)
        return self.observation_to_image(observation), reward, done, info


sokoban_game = SokobanGame()
mcp = FastMCP(name="Sokoban Game Server")

def create_metadata_content(metadata: dict | None = None) -> str:
    """Create metadata string with current game state"""
    if metadata is None:
        metadata = {}
    metadata_str = f"```metadata\n{json.dumps(metadata, indent=2)}\n```"
    return TextContent(type="text", text=metadata_str)

@mcp.tool
def get_init_state() -> List[Union[TextContent, Image]]:
    """Initialize the Sokoban game. Can only be called once."""

    if sokoban_game.initialized:
        return ["Error: Game already initialized. Cannot initialize multiple times.", create_metadata_content({})]
    
    # Extract seed and room_id from headers (with defaults for stdio mode)
    if not ('ENV_SEED' in os.environ and 'ENV_ROOM_ID' in os.environ):
        return ["Error: ENV_SEED and ENV_ROOM_ID must be set.", create_metadata_content({})]
    
    seed = int(os.environ['ENV_SEED'])
    room_id = os.environ.get('ENV_ROOM_ID', 'Sokoban-small-v0')
    stage_width = int(os.environ.get('ENV_STAGE_WIDTH', 5))
    stage_height = int(os.environ.get('ENV_STAGE_HEIGHT', 5))
    num_boxes = int(os.environ.get('ENV_NUM_BOXES', 1))
    num_gen_steps = int(os.environ.get('ENV_NUM_GEN_STEPS', "-1"))
    num_gen_steps = None if num_gen_steps == -1 else num_gen_steps
    
    # Initialize environment
    sokoban_game.initialize(seed, room_id, stage_width, stage_height, num_boxes, num_gen_steps)
    
    # Get initial image
    initial_image = sokoban_game.get_state_image()
    
    description = f"Sokoban game initialized. Ready to play! Use move_sequence to make moves."

    metadata = {
        "room_id": sokoban_game.room_id,
        "seed": sokoban_game.seed,
        "step_count": sokoban_game.step_count,
        "accumulated_reward": sokoban_game.accumulated_reward,
        "done": sokoban_game.done,
        "move_history": sokoban_game.move_history,
        "is_passed": False,
    }
    return [description, initial_image, create_metadata_content(metadata)]

@mcp.tool
def move_sequence(
    directions: List[str] = Field(..., description='List of movement directions, e.g. ["up", "down", "left", "right"] (max 5 moves)'),
    ) -> List[Union[TextContent, Image]]:
    """Execute a sequence of moves in the Sokoban game."""

    if not sokoban_game.initialized:
        return ["Error: Game not initialized. Call get_init_state first.", create_metadata_content()]
    
    if len(directions) > 5:
        return [f"Error: Maximum 5 moves allowed, got {len(directions)} moves.", create_metadata_content()]
    
    if sokoban_game.done:
        return ["Error: Game already completed. Cannot make more moves.", create_metadata_content()]
    
    # Action mapping (push actions)
    action_map = {"up": 1, "down": 2, "left": 3, "right": 4}
    
    results = []
    
    is_passed = False
    for direction in directions:
        try:
            move_image, reward, done, info = sokoban_game.step(direction)
        except Exception as e:
            results.append(f"Error: Invalid direction '{direction}'.")
            break
        
        if done and reward > 0:
            results.append("🎉 Congratulations! You solved the puzzle! 🎉")
            is_passed = True
            break
        elif done:
            results.append("Game completed.")
            break
        
        # Add move result in the requested format
        move_text = f"Step {sokoban_game.step_count}, Move: {direction}"
        results.extend([move_text, move_image])

    metadata = {
        "room_id": sokoban_game.room_id,
        "seed": sokoban_game.seed,
        "step_count": sokoban_game.step_count,
        "accumulated_reward": sokoban_game.accumulated_reward,
        "done": sokoban_game.done,
        "move_history": sokoban_game.move_history,
        "is_passed": is_passed,
    }
    
    # Add metadata at the end
    results.append(create_metadata_content(metadata))
    results = [TextContent(type="text", text=r) if isinstance(r, str) else r for r in results]
    
    return results

if __name__ == "__main__":
    mcp.run(transport="stdio")  # Default transport is stdio