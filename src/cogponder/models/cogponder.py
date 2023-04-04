import torch
from torch import nn
from pytorch_lightning import LightningModule
from ..losses import ResponseLoss, ResponseTimeLoss
# import torchmetrics

from .halting import HaltingModule
from .operator import SimpleOperatorModule
from .recurrence import RecurrenceModule


class CogPonderModel(LightningModule):
    def __init__(
        self,
        task,
        inputs_dim,
        outputs_dim,
        embeddings_dim,
        max_response_step,
        n_contexts=1,
        context_embeddings_dim=4,
        response_loss_beta=1.,
        time_loss_beta=1.,
        learning_rate=1e-3,
        example_input_array=None
    ):
        """CogPonder model written in PyTorch Lightning.

        Args:
            task (str): task to perform
            inputs_dim (int): dimensionality of input data
            outputs_dim (int): dimensionality of output data
            embeddings_dim (int): dimensionality of embeddings
            max_response_step (int): maximum number of response steps
            n_contexts (int): number of contexts (i.e., embeddings). Defaults to 1.
            context_embeddings_dim (int): dimensionality of context embeddings. Defaults to 4.
            response_loss_beta (float): weight of response loss. Defaults to 1.
            time_loss_beta (float): weight of response time loss. Defaults to 1.
            learning_rate (float): learning rate, defaults to 1e-3.
            example_input_array (torch.Tensor): optional example for the input array.

        """

        super().__init__()
        self.save_hyperparameters(ignore=['example_input_array'], logger=False)

        self.task = task
        self.inputs_dim = inputs_dim
        self.outputs_dim = outputs_dim
        self.embeddings_dim = embeddings_dim
        self.max_response_step = max_response_step
        self.n_contexts = n_contexts
        self.response_loss_beta = response_loss_beta
        self.time_loss_beta = time_loss_beta
        self.learning_rate = learning_rate
        self.example_input_array = example_input_array

        # self.train_accuracy = torchmetrics.Accuracy(task='multiclass')
        # self.val_accuracy = torchmetrics.Accuracy(task='multiclass')

        # init losses
        self.resp_loss_fn = ResponseLoss()
        self.time_loss_fn = ResponseTimeLoss(self.max_response_step)

        # init nodes
        self.operator_node = SimpleOperatorModule(self.embeddings_dim, self.outputs_dim)
        self.halt_node = HaltingModule(self.embeddings_dim, self.max_response_step)
        self.recurrence_node = RecurrenceModule(self.inputs_dim, self.embeddings_dim)

        # init contextual embeddings (e.g., subject)
        self.context_embeddings = nn.Embedding(self.n_contexts, self.context_embeddings_dim)

        # init embeddings
        self.embeddings = nn.Embedding(self.n_contexts, self.embeddings_dim)

    def forward(self, x, context_ids=None):

        """CogPonder forward pass

        Args
        ----
            context (torch.Tensor): contextual information of shape (batch_size,)
            x (torch.Tensor): input data of shape (batch_size, inputs_dim)

        Returns
        -------
            (y_steps, p_steps, halt_steps)
        """

        batch_size = x.size(0)

        # append context-specific embeddings to input
        context_features = self.context_embeddings(context_ids)
        x = torch.cat([x, context_features], dim=1)

        # init parameters
        h = self.embeddings(context_ids)
        p_continue = torch.ones(batch_size, device=self.device)
        halt_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        y_list = []  # list of y_steps (step-wise responses)
        p_list = []  # list of p_steps (step-wise halting probabilities)

        # main loop
        for n in range(1, self.max_response_step + 1):

            # 1. operate
            y_step = self.operator_node(h)
            y_list.append(y_step)

            # 2. calculate halting probability
            lambda_n = self.halt_node(h, n)
            p_list.append(p_continue * lambda_n)
            p_continue = p_continue * (1 - lambda_n)
            halt_steps = torch.max(
                halt_steps,
                ((halt_steps == 0) * n * torch.bernoulli(lambda_n)).to(torch.long)
            )

            # 3. loop
            h = self.recurrence_node(x, h)

            # 4. stop if all the samples have halted
            # IGNORE: enable for debugging or stopping the recurrent loop upon halt.
            if False & (halt_steps > 0).sum() == batch_size:
                break

        # prepare outputs
        y_steps = torch.stack(y_list)  # (step, batch)
        p_steps = torch.stack(p_list)  # (step, batch)

        # normalize halting probabilities (sums to 1 across steps)
        for i in range(batch_size):
            halt_step = halt_steps[i] - 1
            p_steps[halt_step:, i] = 0.0
            p_steps[halt_step, i] = 1 - p_steps[:halt_step, i].sum()

        return y_steps, p_steps, halt_steps

    def training_step(self, batch, batch_idx):

        contexts, stimuli, trial_types, responses, rt_true, _ = batch

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(stimuli, contexts)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, responses)
        time_loss = self.time_loss_fn(p_halts, rt_true, logger=self.logger.experiment, step=self.global_step)
        loss = self.response_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # log losses
        self.log('train/resp_loss', resp_loss, on_epoch=True, on_step=False)
        self.log('train/time_loss', time_loss, on_epoch=True, on_step=False)
        self.log('train/total_loss', loss, on_epoch=True, logger=True, on_step=False)

        # compute and log accuracy (assuming binary classification)
        # y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
        # accuracy = (y_pred.int() == y_true.int()).float().mean()
        # self.log('train/accuracy', accuracy, on_epoch=True)
        # self.train_accuracy(y_pred, y_true.int())
        # self.log('train/accuracy', self.train_accuracy, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):

        contexts, stimuli, trial_types, responses, rt_true, _ = batch

        # forward pass
        y_steps, p_halts, rt_pred = self.forward(stimuli, contexts)
        y_pred = torch.argmax(y_steps, dim=-1).gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)

        print(y_steps.shape, p_halts.shape, rt_pred.shape)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, responses)
        time_loss = self.time_loss_fn(p_halts, rt_true)
        loss = self.response_loss_beta * resp_loss + self.time_loss_beta * time_loss

        # log losses
        self.log('val/resp_loss', resp_loss, on_epoch=True, on_step=False)
        self.log('val/time_loss', time_loss, on_epoch=True, on_step=False)
        self.log('val/total_loss', loss, on_epoch=True, logger=True, on_step=False)

        match self.task:
            case 'nback':
                # compute and log accuracy (assuming binary classification)
                # y_pred = y_steps.gather(dim=0, index=rt_pred[None, :] - 1,)[0]  # (batch_size,)
                # accuracy = (y_pred.int() == y_true.int()).float().mean()
                # self.log('val/accuracy', accuracy, on_epoch=True)
                # self.val_accuracy(y_pred, y_true.int())
                # self.log('val/accuracy', self.val_accuracy, on_epoch=True)
                pass
            case 'stroop':
                is_corrects_pred = (y_pred.long() == responses).float()
                cong_is_corrects = torch.where(trial_types == 1, is_corrects_pred, torch.nan)
                incong_is_corrects = torch.where(trial_types == 0, is_corrects_pred, torch.nan)

                accuracy = torch.nanmean(is_corrects_pred)
                cong_accuracy = torch.nanmean(cong_is_corrects)
                incong_accuracy = torch.nanmean(incong_is_corrects)

                self.log('val/accuracy', accuracy, on_epoch=True)
                self.log('val/accuracy_congruent', cong_accuracy, on_epoch=True)
                self.log('val/accuracy_incongruent', incong_accuracy, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
