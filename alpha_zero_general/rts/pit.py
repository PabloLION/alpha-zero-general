from alpha_zero_general import Arena
from alpha_zero_general.rts.rts_game import RTSGame, display
from alpha_zero_general.rts.src.config_class import CONFIG

"""
rts/pit.py

Compares 2 players against each other and outputs num wins p1/ num wins p2/ draws
"""
CONFIG.set_runner("pit")  # set visibility as pit
g = RTSGame()
player1, player2 = CONFIG.pit_args.create_players(g)
arena = Arena.Arena(player1, player2, g, display=display)
print(arena.play_games(CONFIG.pit_args.num_games, verbose=CONFIG.visibility))
