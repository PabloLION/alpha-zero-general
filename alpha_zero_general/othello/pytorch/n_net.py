import os
import time

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from alpha_zero_general.neural_net import NeuralNetInterface
from alpha_zero_general.othello import (
    OthelloBoardDataType,
    OthelloBoardTensor,
    OthelloBooleanBoardTensor,
    OthelloNNArg,
    OthelloPolicyTensor,
    OthelloTrainingExample,
)
from alpha_zero_general.othello.othello_game import OthelloGame
from alpha_zero_general.othello.pytorch.othello_n_net import OthelloNNet
from alpha_zero_general.utils import AverageMeter

args = OthelloNNArg(
    lr=0.001,
    dropout=0.3,
    epochs=10,
    batch_size=64,
    cuda=torch.cuda.is_available(),
    num_channels=512,
)


class OthelloTorchNNInterface(
    NeuralNetInterface[
        OthelloBoardTensor, OthelloBooleanBoardTensor, OthelloPolicyTensor
    ]
):
    def __init__(self, game: OthelloGame) -> None:
        self.nn = OthelloNNet(game, args)
        self.board_x, self.board_y = game.get_board_size()
        self.action_size = game.get_action_size()

        if args.cuda:
            self.nn.cuda()

    def train(self, examples: list[OthelloTrainingExample]) -> None:
        """
        examples: list of examples, each example is of form (board, pi, v)
        """
        optimizer = optim.Adam(self.nn.parameters())

        for epoch in range(args.epochs):
            print("EPOCH ::: " + str(epoch + 1))
            self.nn.train()
            pi_losses = AverageMeter()
            v_losses = AverageMeter()

            batch_count = int(len(examples) / args.batch_size)

            tq = tqdm(range(batch_count), desc="Training Net")
            for _ in tq:
                sample_ids = np.random.randint(len(examples), size=args.batch_size)
                boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
                boards = torch.FloatTensor(np.array(boards).astype(np.float64))
                target_pis = torch.FloatTensor(np.array(pis))
                target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

                # predict
                if args.cuda:
                    boards, target_pis, target_vs = (
                        boards.contiguous().cuda(),
                        target_pis.contiguous().cuda(),
                        target_vs.contiguous().cuda(),
                    )

                # compute output
                out_pi, out_v = self.nn(boards)
                l_pi = self.loss_pi(target_pis, out_pi)
                l_v = self.loss_v(target_vs, out_v)
                total_loss = l_pi + l_v

                # record loss
                pi_losses.update(l_pi.item(), boards.size(0))
                v_losses.update(l_v.item(), boards.size(0))
                tq.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)  # type: ignore # tqdm problem

                # compute gradient and do SGD step
                optimizer.zero_grad()
                total_loss.backward()  # type: ignore # pytorch stubs missing
                optimizer.step()  # type: ignore # pytorch stubs missing

    def predict(self, board: OthelloBoardTensor) -> tuple[OthelloPolicyTensor, float]:
        """
        board: np array with board
        """
        # timing
        time.time()

        # preparing input
        board_torch = torch.FloatTensor(board.astype(OthelloBoardDataType))
        if args.cuda:
            board_torch = board_torch.contiguous().cuda()
        board_torch = board_torch.view(1, self.board_x, self.board_y)
        self.nn.eval()
        with torch.no_grad():
            pi, v = self.nn(board_torch)

        # print('PREDICTION TIME TAKEN : {0:03f}'.format(time.time()-start))
        return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]  # type: ignore # pytorch problem

    def loss_pi(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return -torch.sum(targets * outputs) / targets.size()[0]

    def loss_v(self, targets: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
        return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]

    def save_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        filepath = os.path.join(folder, filename)
        if not os.path.exists(folder):
            print(
                "Checkpoint Directory does not exist! Making directory {}".format(
                    folder
                )
            )
            os.mkdir(folder)
        else:
            print("Checkpoint Directory exists! ")
        torch.save(  # type: ignore # pytorch problem
            {
                "state_dict": self.nn.state_dict(),
            },
            filepath,
        )

    def load_checkpoint(
        self, folder: str = "checkpoint", filename: str = "checkpoint.pth.tar"
    ) -> None:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py#L98
        filepath = os.path.join(folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError("No model in path {}".format(filepath))
        map_location = None if args.cuda else "cpu"
        checkpoint = torch.load(filepath, map_location=map_location)  # type: ignore # pytorch problem
        self.nn.load_state_dict(checkpoint["state_dict"])
