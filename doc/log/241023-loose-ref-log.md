## Summary

I stopped making this because I think it's better to use a more common form
like `pyspiel` or `gym` for the game interface.
In comparison they are:

- more mature / stable
- more widely used
- better documented
- supports incomplete information games
- THO it's harder to setup the dev environment

## Log

### quick fix

- [x] py313_functions  -> py313_backport
- [x] GamePolicyType: TypeAlias = float32
- [x] `self.nnet.model`
- [x] `TicTacToeNNet as onnet`

### modernize

- [x] .no-git
- [x] .TicTacToeNNet
- [x] isort
- [x] util *
- [x] unused imports
- [x] keras types
- [ ] add type arguments for all `np.ndarray` in the project. If needed, merge the similar ones into one single type and put it in type.py.
- [x] `from alpha_zero_general.type import Any`
- [x] Fix deps
  - ImportError while importing test module '/Users/pablo/LocalDocs/repo/eye-on/suragnair/alpha-zero-general/tests/test_all_games.py'.
- [x] Do not write all dependency like we do in requirements. Instead, only control the top-level dependencies via pyproject.toml. The rest should be installed automatically.
- [x] use correct default param like
- [x] Why do we need BoardState
- [x] type protocols and abstract classes
- [ ] project level config
  - [ ] and `from alpha_zero_general.rts.src.config_class import CONFIG`
- `array2string(board)` instead of `board.tostring()`
- better file save paths, remove `os` (security)
- [ ] enum type for `Player` and `Piece`
- [x] rm folder `chore`
- [ ] use pathlib for paths
- [ ] `import sys` and `sys.path.append("..")`
- [x] `np.array2string(board)` as hash has performance issues
- [ ] related to last get str representation
  - __str__ and __repr__ are two methods.
  - to use it as hash, use `__hash__` is better
- [ ] `fmt: off` for data list, like in `test_symmetries_n3`
- [x] style/type-gymnastics
- [x] retire `alpha_zero_general.type`
- [x] rename `./alpha_zero_general/neural_net.py` to `neural_net_interface.py`
- [ ] maybe `game.py` should `Match` that aggregates `Board` and associates with `Rule`.
- [ ] move to py312 for generics(maybe?)
- [ ] merge two nn files (interface wrapper)
- [ ] tests
  - [ ] `dots_and_boxes_test.py`
  - [ ] two `test_connect4.py`
- consider using Generator for ret val of `Game.get_symmetries`
- need a `player` / `agent` class
- split the alpha_zero process with the examples
  - define a protocol that specifies all the classes needed for alpha_zero_general
  - add all the game classes to the protocol, in a separate module, like "examples"
- [ ] add a dictionary for the project
- [ ] for tafl game, TaflGameVariant add `parse board` and `parse pieces`
