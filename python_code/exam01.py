"""Rock-paper-scissors game for Python 3.9."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Optional

try:
	from pyfiglet import figlet_format
except ImportError:  # pragma: no cover - graceful fallback if dependency is absent
	figlet_format = None


RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"

CHOICES = ("rock", "paper", "scissors")
WIN_MAP = {
	"rock": "scissors",
	"paper": "rock",
	"scissors": "paper",
}
CHOICE_ALIASES = {
	"r": "rock",
	"rock": "rock",
	"p": "paper",
	"paper": "paper",
	"s": "scissors",
	"scissor": "scissors",
	"scissors": "scissors",
}
STATS_FILE = Path(__file__).with_name("rps_stats.json")


def load_stats() -> dict:
	"""Load persistent stats from disk."""

	default_stats = {
		"wins": 0,
		"losses": 0,
		"draws": 0,
		"games_played": 0,
		"player_choices": {choice: 0 for choice in CHOICES},
	}

	if not STATS_FILE.exists():
		return default_stats

	try:
		with STATS_FILE.open("r", encoding="utf-8") as file_handle:
			stored_stats = json.load(file_handle)
	except (OSError, json.JSONDecodeError):
		return default_stats

	player_choices = stored_stats.get("player_choices", {})
	normalized_choices = {choice: int(player_choices.get(choice, 0)) for choice in CHOICES}

	return {
		"wins": int(stored_stats.get("wins", 0)),
		"losses": int(stored_stats.get("losses", 0)),
		"draws": int(stored_stats.get("draws", 0)),
		"games_played": int(stored_stats.get("games_played", 0)),
		"player_choices": normalized_choices,
	}


def save_stats(stats: dict) -> None:
	"""Persist stats to disk."""

	with STATS_FILE.open("w", encoding="utf-8") as file_handle:
		json.dump(stats, file_handle, ensure_ascii=False, indent=2)


def print_banner() -> None:
	"""Render the opening title."""

	title = "Rock Paper Scissors"
	if figlet_format is not None:
		print(f"{CYAN}{figlet_format(title)}{RESET}")
	else:
		print(f"{CYAN}{title}{RESET}")


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments."""

	parser = argparse.ArgumentParser(description="Play rock-paper-scissors.")
	parser.add_argument(
		"--difficulty",
		choices=("easy", "hard"),
		default="easy",
		help="Computer strategy difficulty.",
	)
	return parser.parse_args()


def normalize_choice(raw_choice: str) -> Optional[str]:
	"""Validate and normalize a user choice."""

	cleaned_choice = raw_choice.strip().lower()
	if cleaned_choice in CHOICE_ALIASES:
		return CHOICE_ALIASES[cleaned_choice]
	return None


def get_player_choice() -> Optional[str]:
	"""Prompt the player for a move or quit command."""

	raw_choice = input("請輸入 rock/paper/scissors，或輸入 q 離開：")
	cleaned_choice = raw_choice.strip().lower()

	if cleaned_choice in {"q", "quit", "exit"}:
		return None

	choice = normalize_choice(raw_choice)
	if choice is not None:
		return choice

	print(f"{YELLOW}輸入無效，請重新輸入。{RESET}")
	return get_player_choice()


def choose_computer_move(difficulty: str, stats: dict) -> str:
	"""Choose the computer move based on difficulty."""

	if difficulty == "easy":
		return random.choice(CHOICES)

	player_choices = stats["player_choices"]
	most_common_choice = max(CHOICES, key=lambda choice: player_choices[choice])

	if player_choices[most_common_choice] == 0:
		return random.choice(CHOICES)

	favored_move = next(
		move for move, beaten_choice in WIN_MAP.items() if beaten_choice == most_common_choice
	)

	if random.random() < 0.7:
		return favored_move
	return random.choice(CHOICES)


def determine_result(player_choice: str, computer_choice: str) -> str:
	"""Determine the outcome of a single round."""

	if player_choice == computer_choice:
		return "draw"
	if WIN_MAP[player_choice] == computer_choice:
		return "win"
	return "loss"


def colorize_result(result: str) -> str:
	"""Return a colorized label for the round result."""

	if result == "win":
		return f"{GREEN}你贏了{RESET}"
	if result == "loss":
		return f"{RED}你輸了{RESET}"
	return f"{YELLOW}平手{RESET}"


def update_stats(stats: dict, player_choice: str, result: str) -> None:
	"""Update the running counters."""

	stats["games_played"] += 1
	stats["player_choices"][player_choice] += 1

	if result == "win":
		stats["wins"] += 1
	elif result == "loss":
		stats["losses"] += 1
	else:
		stats["draws"] += 1


def show_scoreboard(stats: dict) -> None:
	"""Print the cumulative scoreboard."""

	print(
		f"{MAGENTA}目前戰績："
		f"{stats['wins']} 勝 / {stats['losses']} 負 / {stats['draws']} 平"
		f"（共 {stats['games_played']} 局）{RESET}"
	)


def play_round(difficulty: str, stats: dict) -> bool:
	"""Run one round. Return False when the player quits."""

	player_choice = get_player_choice()
	if player_choice is None:
		return False

	computer_choice = choose_computer_move(difficulty, stats)
	result = determine_result(player_choice, computer_choice)
	update_stats(stats, player_choice, result)

	print(f"你出：{player_choice} | 電腦出：{computer_choice}")
	print(f"結果：{colorize_result(result)}")
	show_scoreboard(stats)
	save_stats(stats)
	return True


def main() -> None:
	"""Program entry point."""

	args = parse_args()
	stats = load_stats()

	print_banner()
	print(f"{CYAN}難度模式：{args.difficulty}{RESET}")
	print(f"{CYAN}輸入 q 可結束遊戲。{RESET}")

	playing = True
	while playing:
		playing = play_round(args.difficulty, stats)

	save_stats(stats)
	print(f"{CYAN}遊戲結束，資料已儲存至 {STATS_FILE.name}{RESET}")


if __name__ == "__main__":
	main()
