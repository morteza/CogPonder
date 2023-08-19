import torch
from torch import nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ..losses import ResponseLoss, ResponseTimeLoss
import torchmetrics.functional as metrics


from .halting import HaltingModule
from .operator import SimpleOperatorModule, SpatioTemporalOperatorModule
from .iterator import IteratorModule


class CogPonderModel(LightningModule):
    def __init__(
        self,
        task,
        inputs_dim,
        outputs_dim,
        embeddings_dim,
        max_response_step,
        n_contexts=1,
        n_subjects=None,
        subject_embeddings_dim=0,
        response_loss_beta=1.,
        time_loss_beta=1.,
        learning_rate=1e-3,
        operator_type='simple',
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
            subject_embeddings_dim (int): dimensionality of subject embeddings. Defaults to 4.
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
        self.n_subjects = n_subjects
        self.subject_embeddings_dim = subject_embeddings_dim if n_subjects is not None else 0
        self.response_loss_beta = response_loss_beta
        self.time_loss_beta = time_loss_beta
        self.learning_rate = learning_rate
        self.example_input_array = example_input_array

        # self.train_accuracy = torchmetrics.Accuracy(task='multiclass')
        # self.val_accuracy = torchmetrics.Accuracy(task='multiclass')

        # init nodes
        # self.operator_input_fc = nn.Linear(self.inputs_dim + self.subject_embeddings_dim,
        #                                    self.embeddings_dim)

        self.halt_node = HaltingModule(self.embeddings_dim,
                                       self.max_response_step)
        self.iterator_node = IteratorModule(self.inputs_dim,
                                            self.embeddings_dim,
                                            self.n_subjects,
                                            self.subject_embeddings_dim)
        match operator_type.lower():
            case 'spatiotemporal':
                self.operator_node = SpatioTemporalOperatorModule(
                    self.inputs_dim, self.outputs_dim, 4, self.embeddings_dim)
                # FIXME use variable space-time embedding sizes
            case _:  # simple
                self.operator_node = SimpleOperatorModule(
                    self.embeddings_dim, self.outputs_dim)

        # init embeddings
        self.embeddings = nn.Embedding(self.n_contexts,
                                       self.embeddings_dim,
                                       dtype=torch.float)

        # init losses
        self.resp_loss_fn = ResponseLoss()
        self.time_loss_fn = ResponseTimeLoss(self.max_response_step)

    def forward(self, x, subject_ids, context_ids):

        """CogPonder forward pass

        Args
        ----
            context (torch.Tensor): contextual information of shape (batch_size,)
            x (torch.Tensor): input data of shape (batch_size, inputs_dim)

        Returns
        -------
            (y_steps, p_steps, halt_steps)
        """

        context_ids = context_ids.int()
        batch_size, seq_len, n_features = x.shape

        # init embedding, h: (batch_size, embeddings_dim)
        h = self.embeddings(context_ids)

        p_continue = torch.ones(batch_size, device=self.device)
        halt_steps = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        y_list = []  # list of y_steps (step-wise responses)
        p_list = []  # list of p_steps (step-wise halting probabilities)

        # main loop
        for n in range(1, self.max_response_step + 1):

            # 1. operate, test, iterate
            y_step = self.operator_node(x, h)
            lambda_n = self.halt_node(h, step=n)
            h = self.iterator_node(x, h, subject_ids)

            y_list.append(y_step)
            p_list.append(p_continue * lambda_n)
            p_continue = p_continue * (1 - lambda_n)
            halt_steps = torch.max(
                halt_steps,
                ((halt_steps == 0) * n * torch.bernoulli(lambda_n)).to(torch.long)
            )

            # 4. stop if all the samples have halted
            # IGNORE: enable for debugging or stopping the recurrent loop upon halt.
            if False & (halt_steps > 0).sum() == batch_size:
                break

        # prepare outputs
        y_steps = torch.stack(y_list)  # (step, batch)
        p_steps = torch.stack(p_list)  # (step, batch)

        # normalize halting probabilities (sums to 1 across steps)
        for i in range(batch_size):
            halt_step = halt_steps[i] - 1  # FIXME
            p_steps[halt_step:, i] = 0.0
            p_steps[halt_step, i] = 1 - p_steps[:halt_step, i].sum()

        # gather response at halting time
        y_pred = torch.argmax(y_steps, dim=-1).gather(dim=0, index=halt_steps[None, :] - 1,)[0]

        return y_pred, y_steps, p_steps, halt_steps.float()

    def training_step(self, batch, batch_idx):

        subject_ids, trial_ids, contexts, stimuli, y_human, rt_human, y_correct = batch

        # FIXME this is a hack to disable contextual embedding
        contexts = torch.zeros_like(contexts)

        # forward pass
        y_pred, y_steps, p_halts, rt_pred = self.forward(stimuli, subject_ids, contexts)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, y_human.long())
        time_loss = self.time_loss_fn(p_halts, rt_human)
        loss = self.response_loss_beta * resp_loss + self.time_loss_beta * time_loss

        accuracy = metrics.accuracy(y_pred.int(), y_correct.int(),
                                    task='multiclass', num_classes=self.outputs_dim)
        imitation_accuracy = metrics.accuracy(y_pred.int(), y_human.int(),
                                              task='multiclass', num_classes=self.outputs_dim)
        rt_smape = metrics.symmetric_mean_absolute_percentage_error(rt_pred, rt_human)

        # log losses
        self.log('train/resp_loss', resp_loss)
        self.log('train/time_loss', time_loss)
        self.log('train/total_loss', loss)
        self.log('train/accuracy', accuracy)
        self.log('train/imitation_accuracy', imitation_accuracy)
        self.log('train/rt_smape', rt_smape)

        return loss

    def validation_step(self, batch, batch_idx):

        subject_ids, trial_ids, contexts, stimuli, y_human, rt_human, y_correct = batch

        # FIXME this is a hack to disable contextual embedding
        contexts = torch.zeros_like(contexts)

        # forward pass
        y_pred, y_steps, p_halts, rt_pred = self.forward(stimuli, subject_ids, contexts)

        # compute losses
        resp_loss = self.resp_loss_fn(p_halts, y_steps, y_human.long())
        time_loss = self.time_loss_fn(p_halts, rt_human)
        loss = self.response_loss_beta * resp_loss + self.time_loss_beta * time_loss

        accuracy = metrics.accuracy(y_pred.int(), y_correct.int(),
                                    task='multiclass', num_classes=self.outputs_dim)
        imitation_accuracy = metrics.accuracy(y_pred.int(), y_human.int(),
                                              task='multiclass', num_classes=self.outputs_dim)
        rt_smape = metrics.symmetric_mean_absolute_percentage_error(rt_pred, rt_human)

        # log losses
        self.log('val/resp_loss', resp_loss)
        self.log('val/time_loss', time_loss)
        self.log('val/total_loss', loss)
        self.log('val/accuracy', accuracy)
        self.log('val/imitation_accuracy', imitation_accuracy)
        self.log('val/rt_smape', rt_smape)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
