import ast
import pandas as pd
import matplotlib.pyplot as plt

def visualize_reward_components(df, show=True, save_path="reward_components.png"):
    """
    Visualize reward components.

    Parameters
    ----------
    df : pd.DataFrame
        Evaluation dataframe containing columns ['t', 'info'].
    show : bool
        If True -> display figure.
        If False -> save figure to save_path.
    save_path : str
        File path used when show=False.
    """

    # --- parse info column ---
    def parse_info(x):
        if isinstance(x, dict):
            return x
        return ast.literal_eval(x)

    parsed_info = df["info"].apply(parse_info)

    # --- extract reward components ---
    reward_components = parsed_info.apply(lambda d: d.get("reward_components", {}))
    df_components = pd.DataFrame(list(reward_components)).fillna(0.0)

    # --- time axis ---
    t = df["current_date"].values

    components = df_components.columns.tolist()
    n = len(components)

    if n == 0:
        raise ValueError("No reward components found in df['info'].")

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n), sharex=True)

    if n == 1:
        axes = [axes]

    for ax, comp in zip(axes, components):
        ax.plot(t, df_components[comp].values)
        ax.set_ylabel(comp)
        ax.grid(True)

    axes[-1].set_xlabel("current_date")
    fig.suptitle("Reward Components Over Time", fontsize=14)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if show:
        plt.show()
    else:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)   # prevent memory accumulation