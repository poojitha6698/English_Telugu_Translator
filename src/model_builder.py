import tensorflow as tf
from tensorflow.keras import layers

# =========================
# CONFIG
# =========================

VOCAB_SIZE = 15000
MAX_LEN = 40

EMBED_DIM = 128
LATENT_DIM = 256
NUM_HEADS = 4


# =========================
# POSITIONAL EMBEDDING
# =========================

@tf.keras.utils.register_keras_serializable()
class PositionalEmbedding(layers.Layer):

    def __init__(
        self,
        sequence_length,
        vocab_size,
        embed_dim,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embed_dim
        )

        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):

        length = tf.shape(inputs)[-1]

        positions = tf.range(
            start=0,
            limit=length,
            delta=1
        )

        embedded_tokens = self.token_embeddings(inputs)

        embedded_positions = self.position_embeddings(
            positions
        )

        return embedded_tokens + embedded_positions

    def get_config(self):

        config = super().get_config()

        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })

        return config


# =========================
# TRANSFORMER ENCODER
# =========================

@tf.keras.utils.register_keras_serializable()
class TransformerEncoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        dense_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential([
            layers.Dense(
                dense_dim,
                activation="relu"
            ),
            layers.Dense(embed_dim)
        ])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads

    def call(
        self,
        inputs
    ):

        attention_output = self.attention(
            inputs,
            inputs
        )

        proj_input = self.layernorm_1(
            inputs + attention_output
        )

        proj_output = self.dense_proj(
            proj_input
        )

        return self.layernorm_2(
            proj_input + proj_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "dense_dim": self.dense_dim,
            "num_heads": self.num_heads
        })

        return config


# =========================
# TRANSFORMER DECODER
# =========================

@tf.keras.utils.register_keras_serializable()
class TransformerDecoder(layers.Layer):

    def __init__(
        self,
        embed_dim,
        latent_dim,
        num_heads,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.dense_proj = tf.keras.Sequential([
            layers.Dense(
                latent_dim,
                activation="relu"
            ),
            layers.Dense(embed_dim)
        ])

        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

    def get_causal_attention_mask(
        self,
        inputs
    ):

        batch_size = tf.shape(inputs)[0]
        sequence_length = tf.shape(inputs)[1]

        i = tf.range(sequence_length)[:, None]
        j = tf.range(sequence_length)

        mask = tf.cast(
            i >= j,
            dtype="int32"
        )

        mask = tf.reshape(
            mask,
            (1, sequence_length, sequence_length)
        )

        mult = tf.concat(
            [
                tf.expand_dims(batch_size, -1),
                tf.constant([1, 1])
            ],
            axis=0
        )

        return tf.tile(
            mask,
            mult
        )

    def call(
        self,
        inputs,
        encoder_outputs
    ):

        causal_mask = self.get_causal_attention_mask(
            inputs
        )

        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=causal_mask
        )

        out_1 = self.layernorm_1(
            inputs + attention_output_1
        )

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs
        )

        out_2 = self.layernorm_2(
            out_1 + attention_output_2
        )

        proj_output = self.dense_proj(
            out_2
        )

        return self.layernorm_3(
            out_2 + proj_output
        )

    def get_config(self):

        config = super().get_config()

        config.update({
            "embed_dim": self.embed_dim,
            "latent_dim": self.latent_dim,
            "num_heads": self.num_heads
        })

        return config


# =========================
# BUILD MODEL
# =========================

def create_transformer():

    encoder_inputs = tf.keras.Input(
        shape=(None,),
        dtype="int64",
        name="encoder_inputs"
    )

    x = PositionalEmbedding(
        MAX_LEN,
        VOCAB_SIZE,
        EMBED_DIM
    )(encoder_inputs)

    encoder_outputs = TransformerEncoder(
        EMBED_DIM,
        LATENT_DIM,
        NUM_HEADS
    )(x)

    decoder_inputs = tf.keras.Input(
        shape=(None,),
        dtype="int64",
        name="decoder_inputs"
    )

    x = PositionalEmbedding(
        MAX_LEN,
        VOCAB_SIZE,
        EMBED_DIM
    )(decoder_inputs)

    x = TransformerDecoder(
        EMBED_DIM,
        LATENT_DIM,
        NUM_HEADS
    )(
        x,
        encoder_outputs
    )

    decoder_outputs = layers.Dense(
        VOCAB_SIZE,
        activation="softmax"
    )(x)

    model = tf.keras.Model(
        [encoder_inputs, decoder_inputs],
        decoder_outputs
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model    