import numpy as np
import gymnasium as gym
from gymnasium import spaces
from os import path
import matplotlib as mpl

mpl.rcParams['animation.embed_limit'] = 500.0

# ---------------------------------------------------------------------------
# Map and constants
# ---------------------------------------------------------------------------
MAP = [
    "+---------+",
    "|R: | : :G|",
    "| : | : : |",
    "| : : : : |",
    "| | : | : |",
    "|Y| : |B: |",
    "+---------+",
]

WINDOW_SIZE = (550, 350)
LOCS = [(0, 0), (0, 4), (4, 0), (4, 3)]

# --- TWEAK YOUR MILAN LANDMARKS HERE ---
# Format: (Scale Factor, X_Offset, Y_Offset)
# Scale > 1.0 makes it bigger, < 1.0 makes it smaller.
# X_Offset: positive = right, negative = left
# Y_Offset: positive = down, negative = up
LANDMARK_TWEAKS = [
    (1.2, 0, -50),   # [0] Politecnico
    (1.5, 0, -50),    # [1] Duomo
    (1.5, 0, 10), # [2] Castello
    (2, 0, 10)     # [3] San Siro
]

# ---------------------------------------------------------------------------
# External Render Cache and Helpers
# ---------------------------------------------------------------------------
_RENDER_CACHE = {
    "window": None,
    "assets_loaded": False,
    "taxi_imgs": None,
    "passenger_img": None,
    "median_horiz": None,
    "median_vert": None,
    "background_img": None,
    "loc_imgs": None, # Stores the 4 Milan landmarks
    "taxi_orientation": 0,
}

def _get_surf_loc(map_loc, cell_size):
    """Maps internal grid coordinates to PyGame screen pixel coordinates."""
    return ((map_loc[1] * 2 + 1) * cell_size[0], (map_loc[0] + 1) * cell_size[1])

def render_taxi_frame(state, desc, lastaction, window_size, locs):
    try:
        import pygame
    except ImportError as e:
        raise DependencyNotInstalled('pygame is not installed') from e

    if _RENDER_CACHE["window"] is None:
        pygame.init()
        pygame.display.set_caption("Taxi - Milan Edition")
        _RENDER_CACHE["window"] = pygame.Surface(window_size)

    window = _RENDER_CACHE["window"]
    cell_size = (window_size[0] / desc.shape[1], window_size[1] / desc.shape[0])

    if not _RENDER_CACHE["assets_loaded"]:
        here = path.join(path.dirname(__file__), '../imgs/custom_taxi')
        load = lambda name: pygame.transform.scale(pygame.image.load(path.join(here, name)), cell_size)

        _RENDER_CACHE["taxi_imgs"] = [load(n) for n in ("cab_front.png", "cab_rear.png", "cab_right.png", "cab_left.png")]
        _RENDER_CACHE["passenger_img"] = load("passenger.png")
        _RENDER_CACHE["background_img"] = load("taxi_background.png")
        
        _RENDER_CACHE["median_horiz"] = [load(n) for n in ("gridworld_median_left.png", "gridworld_median_horiz.png", "gridworld_median_right.png")]
        _RENDER_CACHE["median_vert"] = [load(n) for n in ("gridworld_median_top.png", "gridworld_median_vert.png", "gridworld_median_bottom.png")]
        
        # --- NEW CUSTOM LOADING FOR LANDMARKS ---
        raw_loc_names = ["polimi_pixel.png", "duomo_pixel.png", "castello_pixel.png", "sansiro_pixel.png"]
        _RENDER_CACHE["loc_imgs"] = []
        
        for i, name in enumerate(raw_loc_names):
            scale_factor = LANDMARK_TWEAKS[i][0]
            raw_img = pygame.image.load(path.join(here, name))
            
            # Scale the image based on the cell size AND the custom tweak factor
            new_size = (int(cell_size[0] * scale_factor), int(cell_size[1] * scale_factor))
            scaled_img = pygame.transform.scale(raw_img, new_size)
            _RENDER_CACHE["loc_imgs"].append(scaled_img)
            
        _RENDER_CACHE["assets_loaded"] = True

    # 1. Draw Background and Grid Medians
    for y in range(desc.shape[0]):
        for x in range(desc.shape[1]):
            cell = (x * cell_size[0], y * cell_size[1])
            window.blit(_RENDER_CACHE["background_img"], cell)
            ch = desc[y][x]
            if ch == b"|":
                if y == 0 or desc[y - 1][x] != b"|": window.blit(_RENDER_CACHE["median_vert"][0], cell)
                elif y == desc.shape[0] - 1 or desc[y + 1][x] != b"|": window.blit(_RENDER_CACHE["median_vert"][2], cell)
                else: window.blit(_RENDER_CACHE["median_vert"][1], cell)
            elif ch == b"-":
                if x == 0 or desc[y][x - 1] != b"-": window.blit(_RENDER_CACHE["median_horiz"][0], cell)
                elif x == desc.shape[1] - 1 or desc[y][x + 1] != b"-": window.blit(_RENDER_CACHE["median_horiz"][2], cell)
                else: window.blit(_RENDER_CACHE["median_horiz"][1], cell)

    # 2. Draw Milan Landmarks (With custom tweaks)
    for i, loc_cell in enumerate(locs):
        sx, sy = _get_surf_loc(loc_cell, cell_size)
        img_to_draw = _RENDER_CACHE["loc_imgs"][i]
        
        # Calculate how to center the image if it was scaled up or down
        center_x_offset = (cell_size[0] - img_to_draw.get_width()) / 2
        center_y_offset = (cell_size[1] - img_to_draw.get_height()) / 2
        
        # Apply the user's manual X and Y offsets
        final_x = sx + center_x_offset + LANDMARK_TWEAKS[i][1]
        final_y = sy + center_y_offset + LANDMARK_TWEAKS[i][2]
        
        window.blit(img_to_draw, (final_x, final_y))

    # 3. Draw Passenger and Taxi
    taxi_row, taxi_col = state["row"], state["col"]
    pass_idx = state["pass_idx"]

    if pass_idx < 4:
        window.blit(_RENDER_CACHE["passenger_img"], _get_surf_loc(locs[pass_idx], cell_size))

    if lastaction in (0, 1, 2, 3):
        _RENDER_CACHE["taxi_orientation"] = lastaction

    taxi_location = _get_surf_loc((taxi_row, taxi_col), cell_size)
    taxi_img_to_draw = _RENDER_CACHE["taxi_imgs"][_RENDER_CACHE["taxi_orientation"]]

    window.blit(taxi_img_to_draw, taxi_location)

    return np.transpose(np.array(pygame.surfarray.pixels3d(window)), axes=(1, 0, 2))

def close_taxi_render():
    if _RENDER_CACHE["window"] is None:
        return
    import pygame
    pygame.display.quit()
    pygame.quit()
    _RENDER_CACHE["window"] = None
    _RENDER_CACHE["assets_loaded"] = False

# ---------------------------------------------------------------------------
# High-Level One-Liner Wrapper
# ---------------------------------------------------------------------------
def render_taxi(state, lastaction):
    """Exposes a clean interface to the student class, mapping tuples and actions."""
    if state is None:
        return None
        
    # Maps streamlined actions (0:UP, 1:RIGHT, 2:DOWN, 3:LEFT) 
    # back to legacy orientation indices (0:SOUTH, 1:NORTH, 2:EAST, 3:WEST)
    action_renderer_map = {0: 1, 1: 2, 2: 0, 3: 3}
    mapped_lastaction = action_renderer_map.get(lastaction, lastaction)
    
    # Converts internal tuple state format to the dict required by the legacy code
    render_state_dict = {
        "row": state[0],
        "col": state[1],
        "pass_idx": state[2],
        "dest_idx": state[3]
    }
    
    return render_taxi_frame(
        state=render_state_dict, desc=np.asarray(MAP, dtype="c"), lastaction=mapped_lastaction,
        window_size=WINDOW_SIZE, locs=LOCS
    )