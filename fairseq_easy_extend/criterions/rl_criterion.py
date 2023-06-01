import math
from argparse import Namespace

from typing import Union

import torch
import torch.nn.functional as F

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders
from fairseq.dataclass import FairseqDataclass

from dataclasses import dataclass, field

from fairseq.logging import metrics

from sacrebleu.metrics import BLEU, CHRF
from bert_score import BERTScorer

from comet import download_model, load_from_checkpoint

import wandb


@dataclass
class RLCriterionConfig(FairseqDataclass):
    sentence_level_metric: str = field(
        default="bleu",
        metadata={"help": "sentence level metric"},
    )
    wandb_entity: Union[str, None] = field(
        default=None,
        metadata={"help": "WANBD entity name"},
    )
    wandb_project: str = field(
        default="nlp_machine_translation_nar_rl",
        metadata={"help": "WANBD Project name"},
    )


@register_criterion("rl_loss", dataclass=RLCriterionConfig)
class RLCriterion(FairseqCriterion):
    def __init__(self, task, sentence_level_metric, wandb_project, wandb_entity):
        super().__init__(task)
        self.metric = sentence_level_metric
        self.tokenizer = encoders.build_tokenizer(Namespace(tokenizer="moses"))
        self.tgt_dict = task.target_dictionary
        self.src_dict = task.source_dictionary

        self.bleu = BLEU(effective_order=True)
        self.chrf = CHRF()

        if self.metric == "bert":
            self.bert = BERTScorer(lang="en", rescale_with_baseline=True).score
        elif self.metric == "comet":
            self.comet = load_from_checkpoint(
                download_model("Unbabel/wmt22-comet-da")
            ).predict

        self.run = (
            wandb.init(project=wandb_project, entity=wandb_entity)
            if wandb_entity
            else None
        )

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        # get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(outs, tgt_tokens, src_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach(),
        }
        if self.run is not None:
            wandb.log(logging_output)

        return loss, sample_size, logging_output

    def decode(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            s = self.tokenizer.decode(s)
        return s

    def compute_reward(self, outputs, targets):
        """
        #we take a softmax over outputs
        probs = F.softmax(outputs, dim=-1)
        #argmax over the softmax \ sampling (e.g. multinomial)
        samples_idx = torch.multinomial(probs, 1, replacement=True)
        sample_strings = self.tgt_dict.string(samples_idx)  #see dictionary class of fairseq
        #sample_strings = "I am a sentence"
        reward_vals = evaluate(sample_strings, targets)
        return reward_vals, samples_idx
        """
        pass

    def _compute_loss(self, outputs, targets, sources, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        sources: batch x len
        masks:   batch x len
        """
        bsz = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        # probs = F.softmax(outputs, dim=-1)
        log_probs = F.log_softmax(outputs, dim=-1)

        with torch.no_grad():
            # sample_idx = torch.multinomial(
            #     probs.view(-1, vocab_size), 1, replacement=True
            # ).view(bsz, seq_len)
            sample_idx = torch.multinomial(
                torch.exp(log_probs.view(-1, vocab_size)), 1, replacement=True
            ).view(bsz, seq_len)

            rewards = []
            snts = []
            tgts = []
            data = []

            for idx in range(bsz):
                # if masks is not None:
                #     mask = masks[idx]
                #     sample_sentence = self.decode((sample_idx[idx] * mask).unsqueeze(0))
                #     target_sentence = self.decode((targets[idx] * mask).unsqueeze(0))
                # else:
                sample_sentence = self.decode(sample_idx[idx])
                target_sentence = self.decode(targets[idx])

                if self.metric == "bleu":
                    rewards.append(
                        [
                            self.bleu.sentence_score(
                                sample_sentence, [target_sentence]
                            ).score
                        ]
                        * seq_len
                    )
                elif self.metric == "chrf":
                    rewards.append(
                        [
                            self.chrf.sentence_score(
                                sample_sentence, [target_sentence]
                            ).score
                        ]
                        * seq_len
                    )
                elif self.metric == "bert":
                    tgts.append(target_sentence)
                    snts.append(sample_sentence)
                elif self.metric == "comet":
                    # copy paste of decode but using src_dict this time
                    source_sentence = self.src_dict.string(
                        sources[[idx], :].int().cpu(), "@@ ", "UNKNOWNTOKENINHYP"
                    )
                    # no idea if our tokenizer works on this
                    source_sentence = self.tokenizer.decode(source_sentence)

                    data.append(
                        {
                            "src": source_sentence,
                            "mt": sample_sentence,
                            "ref": target_sentence,
                        }
                    )
                elif self.metric == "constant":
                    rewards.append([1.0] * seq_len)
                else:
                    raise ValueError("invalid metric")

        # TODO: fix bert and comet reward final shape etc.
        if self.metric == "bert":
            rewards = self.bert(snts, tgts)[2]
            rewards = [[score] * seq_len for score in rewards]
        elif self.metric == "comet":
            rewards = self.comet(data, progress_bar=False).scores
            rewards = [[score] * seq_len for score in rewards]

        rewards = torch.Tensor(rewards).to(outputs.device)

        if masks is not None:
            # probs, targets = probs[masks], targets[masks]
            log_probs, targets = log_probs[masks], targets[masks]
            rewards, sample_idx = rewards[masks], sample_idx[masks]

        # log_probs = torch.gather(
        #     torch.log(probs), -1, sample_idx.unsqueeze(1)
        # ).squeeze()
        log_probs = torch.gather(log_probs, -1, sample_idx.unsqueeze(1)).squeeze()
        loss = -log_probs * rewards
        loss, rewards = loss.mean(), rewards.mean()

        return loss, rewards

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum = sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)
