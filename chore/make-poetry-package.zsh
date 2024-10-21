#!/bin/zsh

# Create project and tests directories
mkdir -p alpha_zero_general tests

# Move all main Python files to the project folder
mv Arena.py Coach.py Game.py MCTS.py NeuralNet.py main.py pit.py utils.py alpha_zero_general/

# Move specific project subfolders to the project folder
mv connect4 dotsandboxes gobang othello rts santorini tafl tictactoe tictactoe_3d alpha_zero_general/

# Move test files to tests/ folder
mv test_all_games.py tests/
mv tests/test_*.py tests/

# Clean up __pycache__ directories
find . -name '__pycache__' -type d -exec rm -r {} +

# Move any relevant project metadata to the root
mv README.md LICENSE poetry.lock pyproject.toml requirements.txt setup_env.sh .

echo "Project reorganization complete. Your project is now structured as a Poetry package."
