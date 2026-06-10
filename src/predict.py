import json
import tensorflow as tf

from .model_builder import (
    PositionalEmbedding,
    TransformerEncoder,
    TransformerDecoder
)

MAX_LEN = 40
VOCAB_SIZE = 15000


class Translator:

    def __init__(self):

        # Load trained model
        self.model = tf.keras.models.load_model(
            "artifacts/final_transformer.keras",
            custom_objects={
                "PositionalEmbedding": PositionalEmbedding,
                "TransformerEncoder": TransformerEncoder,
                "TransformerDecoder": TransformerDecoder
            },
            compile=False
        )

        # Load vocabularies
        with open(
            "artifacts/source_vocab.json",
            encoding="utf-8"
        ) as f:

            source_vocab = json.load(f)

        with open(
            "artifacts/target_vocab.json",
            encoding="utf-8"
        ) as f:

            target_vocab = json.load(f)

        # Rebuild vectorizers
        self.source_vectorizer = (
            tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
                output_mode="int",
                output_sequence_length=MAX_LEN
            )
        )

        self.target_vectorizer = (
            tf.keras.layers.TextVectorization(
                max_tokens=VOCAB_SIZE,
                output_mode="int",
                output_sequence_length=MAX_LEN + 1
            )
        )

        self.source_vectorizer.set_vocabulary(
            source_vocab
        )

        self.target_vectorizer.set_vocabulary(
            target_vocab
        )

        self.target_vocab = target_vocab

    def translate(
        self,
        sentence
    ):

        encoder_input = self.source_vectorizer(
            [sentence]
        )

        decoded_sentence = "start"

        for i in range(MAX_LEN):

            tokenized_target = (
                self.target_vectorizer(
                    [decoded_sentence]
                )[:, :-1]
            )

            predictions = self.model.predict(
                [
                    encoder_input,
                    tokenized_target
                ],
                verbose=0
            )

            sampled_index = tf.argmax(
                predictions[0, i, :]
            ).numpy()

            if sampled_index >= len(
                self.target_vocab
            ):
                break

            sampled_token = (
                self.target_vocab[
                    sampled_index
                ]
            )

            if sampled_token == "end":
                break

            decoded_sentence += (
                " " + sampled_token
            )

        return (
            decoded_sentence
            .replace("start", "")
            .strip()
        )