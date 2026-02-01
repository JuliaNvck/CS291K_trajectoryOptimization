import argparse
import itertools as it

import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.collections import PatchCollection
import numpy as np


MAZE_LINE = """
 - - - - 
|s     g|
 - - - - 
"""

MAZE_OPEN = """
 - - - - 
|s      |
         
|       |
         
|       |
         
|      g|
 - - - - 
"""

MAZE_FASTSLOW = """
 - - - - 
|s     g|
   - -   
| |   | |
         
| | | | |
         
|   |   |
 - - - - 
"""

MAZE_FORK = """
 - - - - 
|       |
   - -   
|s|   |g|
   - -   
|       |
 - - - - 
"""

# In this maze with p_success = 1, it's equally good to go right in row 0 or in
# row 1. But with p_success < 1, it's better to use row 0 to avoid getting
# stuck in one of the dead ends.
MAZE_DEADENDS = """
 - - - - - 
|s        |
           
|         |
           
| | | | | |
           
| | | | |g|
 - - - - - 
"""

MAZE_OBSTACLES_EASY = """
 - - - - - - - 
|s            |
   -   -       
| | | | |     |
   -   -   -   
|         | | |
   -   -   -   
| | | | |    g|
 - - - - - - - 
"""

MAZE_WINDING = """
 - - - - - 
|s|   |   |
           
| | | | | |
           
|   |   |g|
 - - - - - 
"""

MAZE_NAMES = ["line", "open", "fastslow", "fork", "deadends", "obstacles_easy", "winding"]
MAZE_BY_NAME = {name: globals()["MAZE_" + name.upper()] for name in MAZE_NAMES}


ACTIONS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]]

# for visualizing Q fn
ACTION_TRIS = [
    None,
    np.array([[0, 0], [-1, -1], [-1, 1]]),
    np.array([[0, 0], [1, -1], [1, 1]]),
    np.array([[0, 0], [-1, -1], [1, -1]]),
    np.array([[0, 0], [-1, 1], [1, 1]]),
]


class Maze:
    def __init__(self, maze_str):
        """Constructor.

        Args:
            maze_str (str): Maze represented as a string with linebreaks,
                following the format shown in examples above.
        """
        m = maze_str.strip("\n").split("\n")
        assert len(m) % 2 == 1
        assert len(m[0]) % 2 == 1
        for i in range(1, len(m)):
            assert len(m[i]) == len(m[0])
        self.rows = len(m) // 2
        self.cols = len(m[0]) // 2
        self.m = np.array([list(s) for s in m])


    def draw(self, V=None, Q=None, fig=None, ax=None, vmin=None, vmax=None):
        """Draws the maze, and optionally a value or Q-function, using Matplotlib.

        The (fig, ax) and (vmin, vmax) optional arguments help you make
        animations or interactive step-through code for each iteration of Value
        Iteration. Example usage is shown in main(). The (fig, ax) returned by
        the first draw() call should be passed to subsequent draw() calls.
        Typically vmin = 0, vmax = 1/(1-gamma) should be used; otherwise the
        colorbar range will change between algorithm iterations.

        Args:
            V (array(S)): State value function to draw.
            Q (array(S, A)): State-action value function to draw.
            fig (matplotlib Figure): Figure in which to overwrite.
            ax (matplotlib Axis): Axis in which to overwrite.
            vmin (float): Minimum value for V/Q colorbar. Typically 0.
            vmax (float): Maximum value for V/Q colorbar. Typically 1/(1-gamma).

        Returns:
            fig, ax: Matplotlib Figure and Axis. Identical to (fig, ax) args if
                supplied, otherwise new ones are constructed.
        """
        if V is not None and Q is not None:
            raise ValueError("Can only draw one of V or Q at a time.")
        has_cbar = V is not None or Q is not None
        if (fig is None) != (ax is None):
            raise ValueError("Must supply both fig and ax, or neither.")

        first = fig is None or ax is None
        if first:
            w = self.cols + has_cbar
            h = self.rows
            fig, ax = plt.subplots(constrained_layout=True, figsize=(w, h), subplot_kw=dict(aspect="equal"))
        else:
            ax.clear()

        # room for thick linewidth
        PAD = 0.1
        ax.set(xlim=(-PAD, 2*self.cols+PAD), ylim=(-PAD, 2*self.rows+PAD))
        # to match +y = down in the ascii art coordinate system
        ax.invert_yaxis()
        ax.axis("off")

        # Build patches for either V or Q
        patch = []
        color = []
        label = None

        if V is not None:
            V = V.reshape((self.rows, self.cols))
            for r in range(self.rows):
                for c in range(self.cols):
                    center = np.array([2 * c + 1, 2 * r + 1])
                    patch.append(patches.Rectangle(center - 1, 2, 2))
                    color.append(V[r, c])
            label = "$V^\\star$"

        elif Q is not None:
            arrows = []
            Q = Q.reshape((self.rows, self.cols, -1))
            for r in range(self.rows):
                for c in range(self.cols):
                    # triangles for the moving actions
                    Qbest = np.max(Q[r, c, :])
                    argbest = np.flatnonzero(np.isclose(Q[r, c, :], Qbest))
                    center = np.array([2 * c + 1, 2 * r + 1])
                    for i, tri in list(enumerate(ACTION_TRIS))[1:]:
                        patch.append(patches.Polygon(center + tri, closed=True))
                        color.append(Q[r, c, i])
                        if i in argbest:
                            trimid = np.mean(tri, axis=0)
                            pts = center + 1.2 * trimid - 0.15 * (tri - trimid)
                            arrows.append(patches.Polygon(pts, closed=True))
                    # circle for stay-put action -- on top of tris
                    patch.append(patches.Circle(center, radius=0.5))
                    color.append(Q[r, c, 0])
                    if 0 in argbest:
                        arrows.append(patches.Circle(center, radius=0.1))
            label = "$Q^\\star$"
            pc_arrows = PatchCollection(arrows, facecolors="white", zorder=100)
            ax.add_collection(pc_arrows)

        if V is not None or Q is not None:
            pc = PatchCollection(
                patch,
                cmap="cool",
                edgecolors="black",
                linewidths=0.25,
            )
            pc.set_array(color)
            if vmin is not None or vmax is not None:
                pc.set_clim(vmin, vmax)
            ax.add_collection(pc)
            if first:
                fig.colorbar(pc, ax=ax, label=label, fraction=1/(self.cols+1))

        kwargs = dict(color="black", markersize=15)

        for r in range(self.rows):
            i = 2 * r + 1
            for c in range(self.cols):
                j = 2 * c + 1
                if self.m[i, j] == "s":
                    ax.plot([j], [i], marker="s", **kwargs)
                if self.m[i, j] == "g":
                    ax.plot([j], [i], marker="*", **kwargs)
                # blocked up or down
                for di in [-1, 1]:
                    lw = 4 if self.m[i + di, j] != " " else 0.25
                    ax.plot([j - 1, j + 1], [i + di, i + di], linewidth=lw, **kwargs)
                # blocked left or right
                for dj in [-1, 1]:
                    lw = 4 if self.m[i, j + dj] != " " else 0.25
                    ax.plot([j + dj, j + dj], [i - 1, i + 1], linewidth=lw, **kwargs)

        return fig, ax


    def to_mdp(self, p_success=1):
        """Converts the maze into a tabular MDP compatible with your Task 1-3 code.

        Your Bellman / VI code should not need to know how we convert between
        1D state indices and 2D row/column indices in the maze.

        Args:
            p_success: Probability of transitioning to the action's ``desired
                state''. See assignment description for details.

        Returns:
            T, r: Transition dynamics and reward tables. See "Assignment-wide
                conventions" docstring in mdpsolve_student.py for more info.
        """
        S = self.rows * self.cols
        A = 5
        reward = np.zeros((S, A))
        T = np.zeros((S, A, S))
        for r in range(self.rows):
            i = 2 * r + 1
            for c in range(self.cols):
                j = 2 * c + 1
                s = r * self.cols + c
                if self.m[i, j] == "g":
                    reward[s, 0] = 1
                # Compute the next state for all actions first, so we can use
                # them in the randomness.
                nextstate = np.zeros(5, dtype=int)
                for a, (di, dj) in enumerate(ACTIONS):
                    if self.m[i + di, j + dj] not in " sg":
                        # blocked - stay in place
                        nextstate[a] = s
                    else:
                        rr, cc = r + di, c + dj
                        if rr < 0 or rr >= self.rows:
                            raise ValueError("maze is not enclosed")
                        if cc < 0 or cc >= self.cols:
                            raise ValueError("maze is not enclosed")
                        nextstate[a] = rr * self.cols + cc
                for a in range(A):
                    for aa in range(A):
                        p = p_success if aa == a else (1 - p_success) / 4
                        T[s, a, nextstate[aa]] += p

        assert np.allclose(T.sum(axis=-1), 1)
        return T, reward


class RobotMaze:
    """A robot with bicycle dynamics inside the maze.

    Public Attributes:
        A_min (array(2)): Minimum value for actions [v, omega].
        A_max (array(2)): Maximum value for actions [v, omega].
        start (array(3)): Initial state (x, y, theta).
    """
    def __init__(self, maze: Maze):
        """Constructs a RobotMaze using the supplied Maze as a map."""
        self.maze = maze
        self.rows = maze.rows
        self.cols = maze.cols
        #                      v     omega
        self.A_min = np.array([0.0, -0.25])
        self.A_max = np.array([0.25, 0.25])

        # Include a wall of unreachable states so we don't need to use special
        # case code for in-bounds checks.
        ncells = (self.rows + 2) * (self.cols + 2)
        reachable = -np.ones((ncells, ncells), dtype=int)

        m = maze.m
        for r in range(self.rows):
            i = 2 * r + 1
            for c in range(self.cols):
                j = 2 * c + 1
                s = self._rc2s(r, c)
                if m[i, j] == "s":
                    self.start = (c + 0.5, r + 0.5, 0)
                if m[i, j] == "g":
                    self._goal = (c + 0.5, r + 0.5, 0)
                reachable[s, s] = 1
                for dr, dc in ACTIONS[1:]:
                    s2 = self._rc2s(r + dr, c + dc)
                    blocked = m[i + dr, j + dc] != " "
                    assert reachable[s, s2] == -1
                    reachable[s, s2] = not blocked

        self._reachable = np.maximum(reachable, 0).astype(bool)
        # drawing
        self._handle_best = None
        self._handles_all = None

    def reward(self, states, actions, sparse=False):
        """Computes the robot-maze MDP reward in parallel.

        Args:
            states (array(N, 3)): (x, y, theta) states.
            actions (array(N, 2)): (v, omega) actions.
            sparse (bool): Enable sparse reward. Not used in Assignment 2.

        Returns:
            rewards (array(N)).
        """
        if not (np.all(actions >= self.A_min[None, :])
            and np.all(actions <= self.A_max[None, :])):
            raise ValueError("Actions do not respect [self.A_min, self.A_max] bounds.")
        dist = np.linalg.norm(states[..., :2] - self._goal[:2], axis=-1)
        if sparse:
            r = 1.0 * (dist <= (0.5 ** 2))
        else:
            r = -dist
        r -= 1e1 * np.sum(actions ** 2, axis=-1)
        return r

    def step(self, states, actions):
        """Computes one step of the robot-maze dynamics in parallel.

        Args:
            states (array(N, 3)): (x, y, theta) states.
            actions (array(N, 2)): (v, omega) actions.

        Returns:
            next_states (array(N, 3)).
        """
        x, y, theta = states.T
        if not (np.all(actions >= self.A_min[None, :])
            and np.all(actions <= self.A_max[None, :])):
            raise ValueError("Actions do not respect [self.A_min, self.A_max] bounds.")
        v, omega = actions.T
        # Possibly in future assignments: Add noise
        # v = v * np.random.uniform(0.9, 1.1, size=v.shape)
        # omega = omega * np.random.uniform(0.9, 1.1, size=omega.shape)
        dx = v * np.cos(theta)
        dy = v * np.sin(theta)
        x2 = x + dx
        y2 = y + dy

        r = np.floor(y).astype(int)
        c = np.floor(x).astype(int)
        s = self._rc2s(r, c)
        r2 = np.floor(y2).astype(int)
        c2 = np.floor(x2).astype(int)
        s2 = self._rc2s(r2, c2)

        reachable = self._reachable[s, s2]
        x2 -= dx * (~reachable)
        y2 -= dy * (~reachable)
        theta2 = theta + omega * reachable

        return np.stack([x2, y2, theta2]).T

    def draw_trajs(self, ax, s_traj_best, s_traj_others):
        """Draws sampling-based trajectory optimization state on the maze.

        Args:
            ax (matplotlib Axes): Axes returned by first call to Maze.draw.
            s_traj_best (array(H, 3)): One state trajectory, to highlight as
                "best" with a heavy red line.
            s_traj_others (array(H, N, 3)):  Many state trajectory samples, to
                draw as faint translucent black lines.
        """
        s_traj_best = s_traj_best.squeeze()
        xb, yb = 2 * s_traj_best[:, :2].T
        xo, yo = 2 * s_traj_others[:, :, :2].transpose((2, 0, 1))
        if self._handle_best is None:
            self._handle_best = ax.plot(xb, yb, linewidth=2, color="red", marker=".", zorder=100)[0]
            self._handles_all = ax.plot(xo, yo, color="black", linewidth=0.5, alpha=0.1)
        self._handle_best.set_xdata(2 * s_traj_best[:, 0])
        self._handle_best.set_ydata(2 * s_traj_best[:, 1])
        for h, x, y in zip(self._handles_all, xo.T, yo.T):
            h.set_xdata(x)
            h.set_ydata(y)

    def _rc2s(self, row, col):  # Private method.
        assert isinstance(row, int) or row.dtype == int
        assert isinstance(col, int) or col.dtype == int
        return (row + 1) * (self.cols + 2) + (col + 1)




if __name__ == "__main__":
    main()
