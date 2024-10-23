## Alpha Zero General (any game, any framework!)

A simplified, highly flexible, commented and (hopefully) easy to understand implementation of self-play based reinforcement learning based on the AlphaGo Zero paper (Silver et al). It is designed to be easy to adopt for any two-player turn-based adversarial game and any deep learning framework of your choice. A sample implementation has been provided for the game of Othello in PyTorch and Keras. An accompanying tutorial can be found [here](https://suragnair.github.io/posts/alphazero.html). We also have implementations for many other games like GoBang and TicTacToe.

To use a game of your choice, subclass the classes in ```Game.py``` and ```NeuralNet.py``` and implement their functions. Example implementations for Othello can be found in ```othello/OthelloGame.py``` and ```othello/{pytorch,keras}/NNet.py```.

```Coach.py``` contains the core training loop and ```MCTS.py``` performs the Monte Carlo Tree Search. The parameters for the self-play can be specified in ```main.py```. Additional neural network parameters are in ```othello/{pytorch,keras}/NNet.py``` (cuda flag, batch size, epochs, learning rate etc.).

To start training a model for Othello:

```bash
python main.py
```

Choose your framework and game in ```main.py```.

### Docker Installation

For easy environment setup, we can use [nvidia-docker](https://github.com/NVIDIA/nvidia-docker). Once you have nvidia-docker set up, we can then simply run:

```bash
./setup_env.sh
```

to set up a (default: pyTorch) Jupyter docker container. We can now open a new terminal and enter:

```bash
docker exec -ti pytorch_notebook python main.py
```

### Experiments

We trained a PyTorch model for 6x6 Othello (~80 iterations, 100 episodes per iteration and 25 MCTS simulations per turn). This took about 3 days on an NVIDIA Tesla K80. The pretrained model (PyTorch) can be found in ```pretrained_models/othello/pytorch/```. You can play a game against it using ```pit.py```. Below is the performance of the model against a random and a greedy baseline with the number of iterations.
![alt tag](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/6x6.png)

A concise description of our algorithm can be found [here](https://github.com/suragnair/alpha-zero-general/raw/master/pretrained_models/writeup.pdf).

### Functionality

This repository provides a flexible and easy-to-understand implementation of self-play based reinforcement learning. It is designed to be adaptable for any two-player turn-based adversarial game and any deep learning framework. The core components include:

- `Game.py`: Base class for defining game rules and logic.
- `NeuralNet.py`: Base class for defining the neural network architecture.
- `Coach.py`: Core training loop.
- `MCTS.py`: Monte Carlo Tree Search implementation.
- `main.py`: Script to start the training process.
- Sample implementations for Othello, GoBang, TicTacToe, Connect4, Dots and Boxes, and more.

### Tests

To ensure the functionality of the repository, tests have been added for various components. The tests cover the following:

- `Arena.py`: Tests for the `Arena` class, including the `playGame` and `playGames` methods.
- `Coach.py`: Tests for the `Coach` class, including the `executeEpisode` and `learn` methods.
- `connect4/Connect4Game.py`: Tests for the `Connect4Game` class, including methods like `getInitBoard`, `getBoardSize`, `getActionSize`, `getNextState`, `getValidMoves`, `getGameEnded`, `getCanonicalForm`, `getSymmetries`, and `stringRepresentation`.
- `dotsandboxes/DotsAndBoxesGame.py`: Tests for the `DotsAndBoxesGame` class, including methods like `getInitBoard`, `getBoardSize`, `getActionSize`, `getNextState`, `getValidMoves`, `getGameEnded`, `getCanonicalForm`, `getSymmetries`, and `stringRepresentation`.
- `gobang/GobangGame.py`: Tests for the `GobangGame` class, including methods like `getInitBoard`, `getBoardSize`, `getActionSize`, `getNextState`, `getValidMoves`, `getGameEnded`, `getCanonicalForm`, `getSymmetries`, and `stringRepresentation`.
- `MCTS.py`: Tests for the `MCTS` class, including the `getActionProb` and `search` methods.
- `NeuralNet.py`: Tests for the `NeuralNet` class, including the `train`, `predict`, `save_checkpoint`, and `load_checkpoint` methods.
- `othello/OthelloGame.py`: Tests for the `OthelloGame` class, including methods like `getInitBoard`, `getBoardSize`, `getActionSize`, `getNextState`, `getValidMoves`, `getGameEnded`, `getCanonicalForm`, `getSymmetries`, and `stringRepresentation`.
- `tictactoe/TicTacToeGame.py`: Tests for the `TicTacToeGame` class, including methods like `getInitBoard`, `getBoardSize`, `getActionSize`, `getNextState`, `getValidMoves`, `getGameEnded`, `getCanonicalForm`, `getSymmetries`, and `stringRepresentation`.

### Citation

If you found this work useful, feel free to cite it as

```
@misc{thakoor2016learning,
  title={Learning to play othello without human knowledge},
  author={Thakoor, Shantanu and Nair, Surag and Jhunjhunwala, Megha},
  year={2016},
  publisher={Stanford University, Final Project Report}
}
```

### Contributing

While the current code is fairly functional, we could benefit from the following contributions:

- Game logic files for more games that follow the specifications in ```Game.py```, along with their neural networks
- Neural networks in other frameworks
- Pre-trained models for different game configurations
- An asynchronous version of the code- parallel processes for self-play, neural net training and model comparison.
- Asynchronous MCTS as described in the paper

Some extensions have been implented [here](https://github.com/kevaday/alphazero-general).

### Contributors and Credits

- [Shantanu Thakoor](https://github.com/ShantanuThakoor) and [Megha Jhunjhunwala](https://github.com/jjw-megha) helped with core design and implementation.

- [Shantanu Kumar](https://github.com/SourKream) contributed TensorFlow and Keras models for Othello.
- [Evgeny Tyurin](https://github.com/evg-tyurin) contributed rules and a trained model for TicTacToe.
- [MBoss](https://github.com/1424667164) contributed rules and a model for GoBang.
- [Jernej Habjan](https://github.com/JernejHabjan) contributed RTS game.
- [Adam Lawson](https://github.com/goshawk22) contributed rules and a trained model for 3D TicTacToe.
- [Carlos Aguayo](https://github.com/carlos-aguayo) contributed rules and a trained model for Dots and Boxes along with a [JavaScript implementation](https://github.com/carlos-aguayo/carlos-aguayo.github.io/tree/master/alphazero).
- [Robert Ronan](https://github.com/rlronan) contributed rules for Santorini.
- [Plamen Totev](https://github.com/plamentotev) contributed Go Text Protocol player for Othello.

Note: Chainer and TensorFlow v1 versions have been removed but can be found prior to commit [2ad461c](https://github.com/suragnair/alpha-zero-general/tree/2ad461c393ecf446e76f6694b613e394b8eb652f).
