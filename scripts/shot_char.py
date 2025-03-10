import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# import csv file
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_FILE = PROJECT_ROOT / 'data' / 'processed' / 'player_positions.csv'


# load csv
def load_shot_data(csv_file):
    df = pd.read_csv(csv_file)
    return df


# Identify what is a shot attempt
# if the ball goes near the rim its a shot
# basket does not move so its a fixed position
def identify_shots(df, hoop_x=25, hoop_y=4):
    # filter for ball positions near the rim
    shot_attempts = df[(df['type'] == 'ball') & (df['shot_status'] == 'shooting')]

    # made or missed (reminder this need to be adjusted if needed)
    shot_attempts['made'] = shot_attempts['y2'].diff().fillna(0) > 0
    return shot_attempts


def draw_court(ax=None):
    if ax is None:
        ax = plt.gca()
    
    ax.plot([0, 50], [0,0], color="black") # baseline
    ax.plot([0,50], [94,94], color="black") # opposite baseline
    ax.plot([0,0], [0,94], color="black") # left sideline
    ax.plot([50, 50], [0,94], color="black") # right sideline


    # Draw the key
    ax.plot([17, 33], [0, 0], color="black") #free throw line
    ax.plot([17, 17], [0, 19], color="black") # left key
    ax.plot([33, 33], [0, 19], color="black") # right key
    ax.plot([17, 33], [19, 19], color="black") #free throw

    # three pt line
    circle = plt.Circle((25, 4), color="black", fill = False)
    ax.add_patch(circle)

    return ax


def plot_shot_chart(shots_df):
    plt.figure(figsize=(10, 9))
    ax = draw_court()


    # makes and misses 
    made_shots = shots_df[shots_df['made']]
    missed_shots = shots_df[~shots_df['made']]


    # plot shots
    sns.scatterplot(x='x2', y='y2', data=made_shots, color="green", label='Made', s=100, marker='o', ax=ax)
    sns.scatterplot(x='x2', y='y2', data=missed_shots, color="red", label='Missed', s=100, marker='X', ax=ax)

    plt.title("Shot Chart - Made vs Missed", frontsize=16)
    plt.xlabel("Court X Coordinate")
    plt.ylabel("Court Y Coordinate")
    plt.legend()
    plt.xlim(0, 50)
    plt.ylim(0, 94)
    plt.grid(False)
    plt.show()

def main():
    df = load_shot_data(CSV_FILE)
    shots_df = identify_shots(df)
    plot_shot_chart(shots_df)


if __name__ == "__main__":
    main()