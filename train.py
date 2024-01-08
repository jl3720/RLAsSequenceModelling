from deeplearning.league import League
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--buffer-path", type=str, default="", help="path to load buffer from")
parser.add_argument("--season", type=int, default=0, help="season to start with")
parser.add_argument("--elo-log-dir", type=str, default=None, help="directory to log elo history")
parser.add_argument("--num-seasons", type=int, default=10, help="number of seasons to train for")

if __name__ == "__main__":
    args = parser.parse_args()
    lea = League(
        path = args.buffer_path,
        season=args.season,
        elo_log_dir=args.elo_log_dir
    )

    for _ in range(args.num_seasons):
        lea.play_season()