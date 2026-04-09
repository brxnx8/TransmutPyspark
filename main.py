import logging
from src.mutation_manager import MutationManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

if __name__ == "__main__":
    MutationManager("config.txt").run()