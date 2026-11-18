"""Tile spawners: random (standard 2048), villain, and helper."""

from spawners.adversarial import HelperSpawner, SearchSpawner, VillainSpawner
from spawners.base import RandomSpawner, Spawner
from spawners.runner import SpawnerGameResult, play_game_with_spawner, play_games_with_spawner

SPAWNERS = {
    "random": RandomSpawner,
    "villain": VillainSpawner,
    "helper": HelperSpawner,
}

__all__ = [
    "SPAWNERS",
    "HelperSpawner",
    "RandomSpawner",
    "SearchSpawner",
    "Spawner",
    "SpawnerGameResult",
    "VillainSpawner",
    "play_game_with_spawner",
    "play_games_with_spawner",
]
