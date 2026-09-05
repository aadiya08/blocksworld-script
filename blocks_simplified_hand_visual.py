import heapq
import itertools
import random
import os
import json
from pathlib import Path
from collections import defaultdict

from PIL import Image, ImageDraw, ImageFont


# Configuration
STARTING_BLOCKS = 3
MAX_BLOCKS = 7
PROBLEMS_PER_LEVEL = 200
OUTPUT_DIR = "blocks_world_dataset"

# State representation: (tuple of stacks, holding block or None)

def normalize_state(stacks, holding):
    """Convert a mutable state into a hashable canonical state."""

    clean_stacks = [
        tuple(stack)
        for stack in stacks
        if stack
    ]

    # Stack ordering does not matter physically, so canonicalize it.
    clean_stacks = tuple(sorted(clean_stacks))

    return (clean_stacks, holding)

def generate_random_state(n, rng=None):
    """Generate a random Blocks World state containing A... blocks."""

    if rng is None:
        rng = random

    blocks = [chr(ord('A') + i) for i in range(n)]

    # Randomly choose an ordering and randomly split it into stacks.
    rng.shuffle(blocks)

    stacks = []

    for block in blocks:
        if not stacks or rng.random() < 0.4:
            stacks.append([block])
        else:
            rng.choice(stacks).append(block)

    return normalize_state(stacks, None)

def get_supports(state):
    """Return {block: support} for a Blocks World state."""

    stacks, holding = state

    supports = {}

    for stack in stacks:
        if not stack:
            continue

        supports[stack[0]] = "TABLE"

        for i in range(1, len(stack)):
            supports[stack[i]] = stack[i - 1]

    # A held block has no support while in the hand.
    if holding is not None:
        supports[holding] = "HAND"

    return supports

def heuristic(current, goal):
    """Number of block-support relationships that are incorrect."""

    current_supports = get_supports(current)
    goal_supports = get_supports(goal)

    incorrect = 0

    for block, goal_support in goal_supports.items():
        if current_supports.get(block) != goal_support:
            incorrect += 1

    return incorrect

def get_possible_moves(state):
    """
    Return a list of:
        (next_state, action)

    where action is one of:
        Pickup(X)
        Putdown(X)
        Stack(X,Y)
        Unstack(X,Y)
    """

    stacks, holding = state

    stacks = [list(stack) for stack in stacks]

    next_states = []

    # ========================================================
    # CASE 1: HAND IS HOLDING A BLOCK
    # ========================================================

    if holding is not None:

        # ----------------------------------------------------
        # Putdown(X)
        # ----------------------------------------------------

        new_stacks = [list(stack) for stack in stacks]
        new_stacks.append([holding])

        next_states.append((
            normalize_state(new_stacks, None),
            f"Putdown({holding})"
        ))

        # ----------------------------------------------------
        # Stack(X,Y)
        # ----------------------------------------------------

        for i, stack in enumerate(stacks):

            if not stack:
                continue

            target = stack[-1]

            new_stacks = [list(s) for s in stacks]
            new_stacks[i].append(holding)

            next_states.append((
                normalize_state(new_stacks, None),
                f"Stack({holding},{target})"
            ))

    # ========================================================
    # CASE 2: HAND IS EMPTY
    # ========================================================

    else:

        for i, stack in enumerate(stacks):

            if not stack:
                continue

            block = stack[-1]

            # ------------------------------------------------
            # Pickup(X)
            #
            # X must be directly on the table, meaning its
            # stack contains only X.
            # ------------------------------------------------

            if len(stack) == 1:

                new_stacks = [list(s) for s in stacks]
                new_stacks[i] = new_stacks[i][:-1]

                next_states.append((
                    normalize_state(new_stacks, block),
                    f"Pickup({block})"
                ))

            # ------------------------------------------------
            # Unstack(X,Y)
            #
            # X is on top of Y.
            # ------------------------------------------------

            else:

                support = stack[-2]

                new_stacks = [list(s) for s in stacks]
                new_stacks[i] = new_stacks[i][:-1]

                next_states.append((
                    normalize_state(new_stacks, block),
                    f"Unstack({block},{support})"
                ))

    return next_states

def a_star(start, goal):
    """
    Run A* from start to goal.

    Returns:
        (states, actions)

    where:
        states  = states along the optimal path
        actions = actions between those states

    Returns None if no solution is found.
    """

    counter = itertools.count()

    start_h = heuristic(start, goal)

    frontier = []

    heapq.heappush(
        frontier,
        (
            start_h,       # f = g + h
            0,             # g
            next(counter),
            start
        )
    )

    # Cheapest known g-value for each state.
    cost_so_far = {
        start: 0
    }

    # came_from[state] = (previous_state, action)
    came_from = {}

    while frontier:

        f, g, _, current = heapq.heappop(frontier)

        # Ignore stale priority-queue entries.
        if g != cost_so_far[current]:
            continue

        # ----------------------------------------------------
        # Goal reached
        # ----------------------------------------------------

        if current == goal:

            states = [current]
            actions = []

            while current in came_from:

                previous, action = came_from[current]

                actions.append(action)
                current = previous
                states.append(current)

            states.reverse()
            actions.reverse()

            return states, actions

        # ----------------------------------------------------
        # Explore successors
        # ----------------------------------------------------

        for next_state, action in get_possible_moves(current):

            new_cost = g + 1

            if (
                next_state not in cost_so_far
                or new_cost < cost_so_far[next_state]
            ):

                cost_so_far[next_state] = new_cost

                h = heuristic(next_state, goal)
                f = new_cost + h

                heapq.heappush(
                    frontier,
                    (
                        f,
                        new_cost,
                        next(counter),
                        next_state
                    )
                )

                came_from[next_state] = (
                    current,
                    action
                )

    return None

def generate_goal_state(n, holding_goal=False):
    """Generate a random goal, optionally with one block in the hand."""
    state = generate_random_state(n)

    if not holding_goal:
        return state

    stacks, _ = state
    stacks = [list(stack) for stack in stacks]
    non_empty = [i for i, stack in enumerate(stacks) if stack]
    i = random.choice(non_empty)
    block = stacks[i].pop()
    return normalize_state(stacks, block)

def generate_problem(n, max_attempts=1000, holding_goal=False):
    """Generate one solvable random problem and solve it optimally."""

    for attempt in range(max_attempts):

        initial = generate_random_state(n)
        goal = generate_goal_state(n, holding_goal)

        if initial == goal:
            continue

        result = a_star(initial, goal)

        if result is not None:

            states, actions = result

            plan_length = len(actions)

            return {
                "initial": initial,
                "goal": goal,
                "states": states,
                "plan": actions,
                "plan_length": plan_length,
                "num_blocks": n
            }

    raise RuntimeError(
        f"Could not generate a solvable problem "
        f"after {max_attempts} attempts."
    )

def generate_dataset(
    starting_blocks=3,
    num_levels=10,
    problems_per_level=100
):
    """
    Generate the Blocks World dataset.

    Returns:
        buckets[k] = list of problems whose optimal plan
                     has length k.
    """

    buckets = defaultdict(list)

    n = starting_blocks

    for level in range(1, num_levels + 1):

        for j in range(1, problems_per_level + 1):

            # Alternate goal hand state so both even and odd
            # solution lengths can occur.
            holding_goal = (j % 2 == 0)

            problem = generate_problem(
                n,
                holding_goal=holding_goal
            )

            problem["level"] = level

            buckets[
                problem["plan_length"]
            ].append(problem)

        n = min(n + 1, MAX_BLOCKS)

    return buckets

def get_font(size=32):
    try:
        return ImageFont.truetype(
            "arial.ttf",
            size
        )
    except:
        return ImageFont.load_default()

def draw_blocks_world(state, title, width=800, height=500):
    """Render one Blocks World state using Pillow."""

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    title_font = get_font(32)
    block_font = get_font(24)

    draw.text((20, 20), title, fill="black", font=title_font)

    stacks, holding = state

    pastel_colors = [
        (255, 179, 186),
        (186, 225, 255),
        (186, 255, 201),
        (255, 223, 186),
        (218, 186, 255),
        (255, 255, 186),
        (186, 255, 255),
        (255, 204, 229),
    ]

    block_width = 90
    block_height = 55
    table_y = height - 70

    # Table
    draw.line(
        (40, table_y, width - 40, table_y),
        fill="black",
        width=4
    )

    # Center all stacks on the table
    gap = 30
    total_width = len(stacks) * block_width + max(0, len(stacks) - 1) * gap
    x_start = max(50, (width - total_width) // 2)

    for stack_index, stack in enumerate(stacks):
        x = x_start + stack_index * (block_width + gap)

        for level, block in enumerate(stack):
            y = table_y - (level + 1) * block_height

            draw.rectangle(
                (x, y, x + block_width, y + block_height),
                fill=pastel_colors[
                    (ord(block) - ord("A")) % len(pastel_colors)
                ],
                outline="black",
                width=3
            )

            bbox = draw.textbbox((0, 0), block, font=block_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            draw.text(
                (
                    x + (block_width - text_width) / 2,
                    y + (block_height - text_height) / 2
                ),
                block,
                fill="black",
                font=block_font
            )

    # Draw the held block visually instead of writing "Hand: X"
    if holding is not None:
        hand_x = width // 2 - block_width // 2
        hand_y = 105
        rope_x = hand_x + block_width // 2

        # Vertical line showing the block being held
        draw.line(
            (rope_x, 45, rope_x, hand_y),
            fill="black",
            width=4
        )

        # Held block
        draw.rectangle(
            (
                hand_x,
                hand_y,
                hand_x + block_width,
                hand_y + block_height
            ),
            fill=pastel_colors[
                (ord(holding) - ord("A")) % len(pastel_colors)
            ],
            outline="black",
            width=3
        )

        bbox = draw.textbbox((0, 0), holding, font=block_font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        draw.text(
            (
                hand_x + (block_width - text_width) / 2,
                hand_y + (block_height - text_height) / 2
            ),
            holding,
            fill="black",
            font=block_font
        )

    return image

def state_to_jsonable(state):
    stacks, holding = state

    return {
        "stacks": [
            list(stack)
            for stack in stacks
        ],
        "holding": holding
    }

def run_full_dataset():
    full_buckets = generate_dataset(
        starting_blocks=STARTING_BLOCKS,
        num_levels=NUM_LEVELS,
        problems_per_level=PROBLEMS_PER_LEVEL
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for k in sorted(full_buckets):
        bucket_dir = output_dir / f"k_{k}"
        bucket_dir.mkdir(exist_ok=True)

        for i, problem in enumerate(full_buckets[k], start=1):
            problem_dir = bucket_dir / f"problem_{i:03d}"
            problem_dir.mkdir(exist_ok=True)

            initial_img = draw_blocks_world(
                problem["initial"],
                "Initial State"
            )
            goal_img = draw_blocks_world(
                problem["goal"],
                "Goal State"
            )

            initial_img.save(problem_dir / "initial.png")
            goal_img.save(problem_dir / "goal.png")

            metadata = {
                "problem_id": f"problem_{i:03d}",
                "level": problem["level"],
                "num_blocks": problem["num_blocks"],
                "plan_length": problem["plan_length"],
                "plan": problem["plan"]
            }

            with open(problem_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

def run_full_dataset():
    buckets = generate_dataset(
        starting_blocks=STARTING_BLOCKS,
        num_levels=MAX_BLOCKS - STARTING_BLOCKS + 1,
        problems_per_level=PROBLEMS_PER_LEVEL
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    for k, problems in sorted(buckets.items()):
        bucket_dir = output_dir / f"k_{k}"
        bucket_dir.mkdir(exist_ok=True)

        for i, problem in enumerate(problems, start=1):
            problem_dir = bucket_dir / f"problem_{i:03d}"
            problem_dir.mkdir(exist_ok=True)

            draw_blocks_world(
                problem["initial"], "Initial State"
            ).save(problem_dir / "initial.png")

            draw_blocks_world(
                problem["goal"], "Goal State"
            ).save(problem_dir / "goal.png")

            metadata = {
                "problem_id": f"problem_{i:03d}",
                "level": problem["level"],
                "num_blocks": problem["num_blocks"],
                "plan_length": problem["plan_length"],
                "plan": problem["plan"]
            }

            with open(problem_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    run_full_dataset()
