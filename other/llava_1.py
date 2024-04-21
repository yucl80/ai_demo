from matplotlib import pyplot as plt
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import ctypes
import array
import numpy as np
import time
import llama_cpp


def get_image_embedding(llm: Llama, llava: Llava15ChatHandler, path: str) -> np.array:
    # Important, otherwise embeddings seem to leak across different
    # invocations of get_image_embedding
    llm.reset()
    llm._ctx.kv_cache_clear()

    image_bytes = llava.load_image(path)

    data_array = array.array("B", image_bytes)
    c_ubyte_ptr = (ctypes.c_ubyte * len(data_array)).from_buffer(data_array)

    t_start = time.time_ns()

    embed = llava._llava_cpp.llava_image_embed_make_with_bytes(
        ctx_clip=llava.clip_ctx,
        image_bytes=c_ubyte_ptr,
        n_threads=6,
        image_bytes_length=len(image_bytes),
    )

    t_embed_llava = time.time_ns()

    n_past = ctypes.c_int(llm.n_tokens)
    n_past_p = ctypes.pointer(n_past)

    # Write the image represented by embed into the llama context
    llava._llava_cpp.llava_eval_image_embed(
        ctx_llama=llm.ctx,
        embed=embed,
        n_batch=llm.n_batch,
        n_past=n_past_p,
    )

    t_eval = time.time_ns()

    print(n_past.value)

    assert llm.n_ctx() >= n_past.value
    llm.n_tokens = n_past.value
    llava._llava_cpp.llava_image_embed_free(embed)

    # Get the embedding out of the LLM
    embedding = np.array(
        llama_cpp.llama_get_embeddings(llm._ctx.ctx)[
            : llama_cpp.llama_n_embd(llm._model.model)
        ]
    )

    t_embed_llm = time.time_ns()

    print(
        f"Total: {float(t_embed_llm - t_start) / 1e6:.2f}ms, LLaVA embed: {float(t_embed_llava - t_start) / 1e6:.2f}ms, LLaMA eval: {float(t_eval - t_embed_llava) / 1e6:.2f}ms, LLaMA embed: {float(t_embed_llm - t_eval) / 1e6:.2f}ms"
    )

    llm.reset()
    return embedding


chat_handler = Llava15ChatHandler(
    clip_model_path="../llava/mmproj-model-f16.gguf", verbose=True
)
llm = Llama(
    model_path="../llava/ggml-model-q5_k.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,
    n_batch=1024,
    logits_all=True,
    n_threads=6,
    offload_kqv=True,
    n_gpu_layers=64,
    embedding=True,
    verbose=True,
)

picture_embed = get_image_embedding(llm, chat_handler, "file:///....jpg")

urls = {
    "cat": "https://i.imgur.com/Iden4uN.png",
    "football": "https://i.imgur.com/Uz8gkcJ.jpeg",
    "firetruck": "https://i.imgur.com/st51tjm.png",
    "microsoft": "https://i.imgur.com/hPYEf5a.jpeg",
}


def get_similarity_matrix(left: np.array, right: np.array):
    return np.dot(left, right.T) / (
        np.linalg.norm(left, axis=1)[:, np.newaxis]
        * np.linalg.norm(right, axis=1)[np.newaxis, :]
    )


embeddings = {
    name: get_image_embedding(llm, chat_handler, url) for name, url in urls.items()
}
image_embedding_matrix = np.array([embeddings[n] for n in sorted(urls.keys())])
image_sim = get_similarity_matrix(image_embedding_matrix, image_embedding_matrix)


def plot_similarity_matrix(
    sim: np.array, x_labels: list[str], y_labels: list[str], **kwargs
):
    fig = plt.figure(**kwargs)
    ax = plt.gca()
    im = ax.imshow(sim, cmap="Wistia")

    ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
    ax.set_yticks(np.arange(len(y_labels)), labels=y_labels)
    for i in range(len(x_labels)):
        for j in range(len(y_labels)):
            ax.text(i, j, f"{sim[j, i]:.2f}", ha="center", va="center", color="k")
    fig.tight_layout()


plot_similarity_matrix(
    image_sim, list(sorted(urls.keys())), list(sorted(urls.keys())), figsize=(5, 3)
)

queries = [
    "funniest meme",
    "picture of men walking",
    "4chan greentext",
    "are you fucking sorry",
    "apple competitors",
    "cartoon animal lying down",
    "woman comforting man",
    "microsoft",
]
