import argparse

import maze
import a2_student as student


def main():
    parser = argparse.ArgumentParser(description="Visualize robot-maze trajectory optimization.")
    parser.add_argument(
        "--maze",
        choices=maze.MAZE_NAMES,
        default="obstacles_easy",
        help="Example maze to use. See top of maze.py for layouts.",
    )
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--variance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    maze_str = maze.MAZE_BY_NAME[args.maze]
    m = maze.Maze(maze_str)
    fig, ax = m.draw()
    rm = maze.RobotMaze(m)

    kwargs = {**vars(args)}
    del kwargs["maze"]
    abest = student.trajopt(rm, **kwargs, plot_ax=ax)

    fig.savefig(f"trajopt_{args.maze}.pdf")


# NOTES:
#
# The default parameters in this file are the same ones used to test your code
# on Gradescope, but the autograder is forgiving: it only checks that the
# trajectory ends in the maze goal cell. This lets us keep samples and iters
# small so the autograder runs quickly.
#
# After you pass the autograder, try experimenting with the algorithm
# parameters to reach the goal (center of the cell) more precisely, or try
# algorithm modifications such as a decaying variance.
#
# Try some other mazes besides obstacles_easy to see the limitations of
# sampling-based trajectory optimization!


if __name__ == "__main__":
    main()
