from modules import *


def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLP Team 3 - Hansoldeco")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="Encrypted Config File Path",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=2,
        help="GPU number you want to use",
    )
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = json.load(f)
    main(config)